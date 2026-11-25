"""Export a Matplotlib segmentation grid as individual PNGs and editable slide objects."""

from io import BytesIO

import numpy as np
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from PIL import Image
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Inches, Pt


def _rgb(color):
    return RGBColor.from_string(to_hex(color)[1:].upper())


def _remove_shadow(shape):
    shape.shadow.inherit = False
    # Some renderers still apply the theme effect when the local effect list is empty.
    for effect_ref in shape._element.xpath("p:style/a:effectRef"):
        effect_ref.set("idx", "0")


def add_figure_slide(deck, fig):
    """Keep the figure's physical dimensions and reuse its scatter rendering and layout."""
    width, height = fig.get_size_inches()
    deck.slide_width = Inches(width)
    deck.slide_height = Inches(height)
    slide = deck.slides.add_slide(deck.slide_layouts[6])

    # Rasterize just the transparent point layers. Captions and rules near a cell
    # boundary must not be baked into the PNG as well as drawn on the slide.
    overlays = [a for a in (*fig.texts, *fig.artists) if a.get_visible()]
    for artist in overlays:
        artist.set_visible(False)
    fig.patch.set_alpha(0)
    fig.set_dpi(350)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pixels = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()))
    for ax in fig.axes:
        bbox = ax.get_window_extent(renderer)
        crop = (round(bbox.x0), pixels.height - round(bbox.y1),
                round(bbox.x1), pixels.height - round(bbox.y0))
        x, y, w, h = ax.get_position().bounds
        with BytesIO() as png:
            pixels.crop(crop).save(png, format="PNG")
            picture = slide.shapes.add_picture(
                png, Inches(x * width), Inches((1 - y - h) * height),
                width=Inches(w * width), height=Inches(h * height),
            )
        picture.name = ax.get_label()
    pixels.close()
    for artist in overlays:
        artist.set_visible(True)

    for line in fig.artists:
        if not isinstance(line, Line2D) or not line.get_visible():
            continue
        (x1, y1), (x2, y2) = line.get_xydata()
        shape = slide.shapes.add_connector(
            MSO_CONNECTOR.STRAIGHT, Inches(x1 * width), Inches((1 - y1) * height),
            Inches(x2 * width), Inches((1 - y2) * height),
        )
        shape.line.color.rgb = _rgb(line.get_color())
        shape.line.width = Pt(line.get_linewidth())
        _remove_shadow(shape)

    for text in fig.texts:
        if not text.get_visible():
            continue
        # Use the actual glyph bounds for placement, with a little room for
        # PowerPoint's font metrics. All shape geometry here is in inches.
        box = text.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
        mid_x, mid_y = (box.x0 + box.x1) / 2, height - (box.y0 + box.y1) / 2
        size = text.get_fontsize()
        color = _rgb(text.get_color())
        if text.get_text() == "●":
            diameter = size * 0.80 / 72
            dot = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(mid_x - diameter / 2),
                                        Inches(mid_y - diameter / 2), Inches(diameter), Inches(diameter))
            dot.fill.solid()
            dot.fill.fore_color.rgb = color
            dot.line.fill.background()
            _remove_shadow(dot)
            continue

        align = text.get_ha()
        extra = 0.12
        left = box.x0 - {"center": extra / 2, "right": extra}.get(align, 0)
        box_height = size * 1.5 / 72
        shape = slide.shapes.add_textbox(Inches(left), Inches(mid_y - box_height / 2),
                                         Inches(box.width + extra), Inches(box_height))
        shape.name = text.get_text()
        _remove_shadow(shape)
        frame = shape.text_frame
        frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
        frame.auto_size = MSO_AUTO_SIZE.NONE
        frame.word_wrap = False
        frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        paragraph = frame.paragraphs[0]
        paragraph.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}[align]
        paragraph.text = text.get_text()
        paragraph.font.name = "Arial"
        paragraph.font.size = Pt(size)
        paragraph.font.bold = text.get_fontweight() == "bold"
        paragraph.font.color.rgb = color
