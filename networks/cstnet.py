"""Original CSTNet downstream models with the legacy state_dict layout.

CrossAttention_Cls and CrossAttention_Seg take the original four tensors.
Their Stage 2 adapters consume packed legacy attributes with measured edge
labels. Four-type weights have no channel for the fifth primitive.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from networks.utils import index_points, knn, square_distance


class _PointMLP(nn.Module):
    """The original 1x1 convolutions, with dropout only on the hidden layer."""

    def __init__(self, dim, inputs, hidden, outputs):
        super().__init__()
        conv, norm, dropout = (
            (nn.Conv1d, nn.BatchNorm1d, nn.Dropout1d) if dim == 1
            else (nn.Conv2d, nn.BatchNorm2d, nn.Dropout2d)
        )
        self.linear_layers = nn.ModuleList([conv(inputs, hidden, 1)])
        self.batch_normals = nn.ModuleList([norm(hidden)])
        self.drop_outs = nn.ModuleList([dropout(0.4)])
        self.outlayer = conv(hidden, outputs, 1)

    def forward(self, x):
        x = F.relu(self.batch_normals[0](self.linear_layers[0](x)))
        return self.outlayer(self.drop_outs[0](x))


class _PointAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fai = _PointMLP(2, channels, channels + 8, channels)
        self.psi = _PointMLP(2, channels, channels + 8, channels)
        self.alpha = _PointMLP(2, channels, channels + 8, channels)
        self.gamma = _PointMLP(2, channels, channels + 8, channels)

    def forward(self, centers, neighbors):
        difference = self.fai(centers) - self.psi(neighbors)
        values = self.alpha(neighbors)
        weights = neighbors.size(1) * self.gamma(difference).softmax(dim=1)
        return ((weights * values).sum(dim=2) / neighbors.size(2)).transpose(1, 2)


@torch.no_grad()
def _sample_centers(xyz, count, segmentation, training):
    batch, points, _ = xyz.shape
    # The classifier drew its initial point on the CPU even for CUDA inputs.
    if not segmentation:
        farthest = torch.randint(points, (batch,), device="cpu").to(xyz.device)
    elif training:
        farthest = torch.randint(points, (batch,), device=xyz.device)
    else:
        farthest = (xyz - xyz.mean(dim=1, keepdim=True)).square().sum(-1).argmax(-1)
    samples = torch.empty(batch, count, dtype=torch.long, device=xyz.device)
    distance = xyz.new_full((batch, points), 1e10)
    rows = torch.arange(batch, device=xyz.device)
    for i in range(count):
        samples[:, i] = farthest
        delta = xyz - xyz[rows, farthest].unsqueeze(1)
        distance = torch.minimum(distance, delta.square().sum(-1))
        farthest = distance.argmax(-1)
    return samples


class _SetAbstraction(nn.Module):
    def __init__(self, n_center, n_near, inputs, widths, segmentation):
        super().__init__()
        self.n_center = n_center
        self.n_near = n_near
        self.segmentation = segmentation
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        for width in widths:
            self.mlp_convs.append(nn.Conv2d(inputs, width, 1))
            self.mlp_bns.append(nn.BatchNorm2d(width))
            inputs = width
        # sa5 checkpoints also contain this block, although global pooling skips it.
        self.attention_points = _PointAttention(widths[-1])
        if n_center is None:
            self.attention_points.requires_grad_(False)

    def forward(self, xyz, direction, edge, primitive, features):
        if self.n_center is None:
            group = torch.cat((xyz, direction, edge, edge, primitive, primitive, features), -1)
            group = group.unsqueeze(1).permute(0, 3, 2, 1)
            for conv, norm in zip(self.mlp_convs, self.mlp_bns):
                group = F.relu(norm(conv(group)))
            pooled = group.max(2).values.transpose(1, 2)
            if self.segmentation:
                pooled = torch.cat((features.max(1, keepdim=True).values, pooled), -1)
            return xyz.new_zeros((xyz.size(0), 1, 3)), None, None, None, pooled

        samples = _sample_centers(xyz, self.n_center, self.segmentation, self.training)
        k = min(self.n_near, xyz.size(1) - 1)
        if k:
            with torch.autocast(xyz.device.type, enabled=False):
                neighbors = knn(xyz.float(), k, is_include_self=False)
        else:
            neighbors = samples.new_zeros((xyz.size(0), 1, 1))
        neighbors = index_points(neighbors, samples)
        center_xyz, center_dir, center_edge, center_prim, center_features = (
            index_points(tensor, samples) for tensor in (xyz, direction, edge, primitive, features)
        )
        near_count = neighbors.size(-1)
        group = torch.cat((
            index_points(xyz, neighbors) - center_xyz.unsqueeze(2),
            index_points(direction, neighbors) - center_dir.unsqueeze(2),
            index_points(edge, neighbors), center_edge.unsqueeze(2).expand(-1, -1, near_count, -1),
            index_points(primitive, neighbors), center_prim.unsqueeze(2).expand(-1, -1, near_count, -1),
            index_points(features, neighbors),
        ), dim=-1).permute(0, 3, 2, 1)
        center = torch.cat((
            torch.zeros_like(center_xyz), torch.zeros_like(center_dir),
            center_edge, center_edge, center_prim, center_prim, center_features,
        ), dim=-1).transpose(1, 2).unsqueeze(2)

        # The segmentation encoder updates BN once on the concatenated group.
        # Classification uses the same BN twice, first on neighbors, then centers.
        if self.segmentation:
            group = torch.cat((center, group), dim=2)
        for conv, norm in zip(self.mlp_convs, self.mlp_bns):
            group = F.relu(norm(conv(group)))
            if not self.segmentation:
                center = F.relu(norm(conv(center)))
        if self.segmentation:
            center, group = group[:, :, :1], group[:, :, 1:]
        attended = self.attention_points(center, group)
        return center_xyz, center_dir, center_edge, center_prim, torch.cat((center_features, attended), -1)


class _FeaturePropagation(nn.Module):
    def __init__(self, inputs, widths):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        for width in widths:
            self.mlp_convs.append(nn.Conv1d(inputs, width, 1))
            self.mlp_bns.append(nn.BatchNorm1d(width))
            inputs = width

    def forward(self, fine_xyz, coarse_xyz, skip, coarse):
        if coarse_xyz.size(1) == 1:
            interpolated = coarse.expand(-1, fine_xyz.size(1), -1)
        else:
            # Preserve the legacy sort/tie order and additive epsilon under AMP.
            with torch.autocast(fine_xyz.device.type, enabled=False):
                distances = square_distance(fine_xyz.float(), coarse_xyz.float()).clamp_min(0)
                distances, indices = distances.sort(dim=-1)
                inverse = (distances[..., :3] + 1e-8).reciprocal()
                weights = inverse / inverse.sum(dim=-1, keepdim=True)
                interpolated = (index_points(coarse, indices[..., :3]) * weights.unsqueeze(-1)).sum(2)
        features = torch.cat((skip, interpolated), -1).transpose(1, 2)
        for conv, norm in zip(self.mlp_convs, self.mlp_bns):
            features = F.relu(norm(conv(features)))
        return features.transpose(1, 2)


class _CstNetEncoder(nn.Module):
    def __init__(self, n_metatype, segmentation):
        super().__init__()
        self.n_metatype = n_metatype
        if n_metatype not in (4, 5):
            raise ValueError("CSTNet supports 4 or 5 primitive channels")
        self.preprocess = _PointMLP(1, 8 + n_metatype, 16, 32)
        extra = 10 + 2 * n_metatype  # relative XYZ/direction, paired edge and type
        self.sa1 = _SetAbstraction(1024, 32, 32 + extra, (64, 72, 80), segmentation)
        self.sa2 = _SetAbstraction(512, 32, 112 + extra, (96, 112, 128), segmentation)
        self.sa3 = _SetAbstraction(128, 64, 240 + extra, (160, 176, 192), segmentation)
        self.sa4 = _SetAbstraction(64, 64, 432 + extra, (256, 320, 384), segmentation)
        self.sa5 = _SetAbstraction(None, None, 816 + extra, (512, 640, 768), segmentation)

    def encode(self, xyz, direction, edge, primitive):
        features = self.preprocess(torch.cat((xyz, direction, edge, primitive), -1).transpose(1, 2))
        level = (xyz, direction, edge, primitive, features.transpose(1, 2))
        skips = [(xyz, level[-1])]
        for layer in (self.sa1, self.sa2, self.sa3, self.sa4, self.sa5):
            level = layer(*level)
            skips.append((level[0], level[-1]))
        return skips


class CrossAttention_Cls(_CstNetEncoder):
    """Legacy four-input classifier, including the original parameter names."""

    def __init__(self, n_classes, n_metatype=5):
        super().__init__(n_metatype, segmentation=False)
        self.fc1 = nn.Linear(768, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, n_classes)

    def forward(self, xyz, eula_angle, edge_nearby, meta_type):
        x = self.encode(xyz, eula_angle, edge_nearby, meta_type)[-1][1].squeeze(1)
        for fc, bn, drop in ((self.fc1, self.bn1, self.drop1), (self.fc2, self.bn2, self.drop2)):
            x = fc(x)
            if self.training and x.size(0) == 1:
                # The current loader keeps the last, possibly singleton, batch.
                x = F.batch_norm(x, bn.running_mean, bn.running_var, bn.weight, bn.bias, eps=bn.eps)
            else:
                x = bn(x)
            x = drop(F.relu(x))
        return F.log_softmax(self.fc3(x), dim=-1)


class CrossAttention_Seg(_CstNetEncoder):
    """Legacy four-input part segmenter."""

    def __init__(self, n_classes, n_metatype=5):
        super().__init__(n_metatype, segmentation=True)
        self.fp5 = _FeaturePropagation(1584 + 816, (768, 640))
        self.fp4 = _FeaturePropagation(640 + 432, (576, 544))
        self.fp3 = _FeaturePropagation(544 + 240, (512, 480))
        self.fp2 = _FeaturePropagation(480 + 112, (384, 320))
        self.fp1 = _FeaturePropagation(320 + 32, (288, 256))
        self.afterprocess = _PointMLP(1, 256, 128, n_classes)

    def forward(self, xyz, eula_angle, edge_nearby, meta_type):
        levels = self.encode(xyz, eula_angle, edge_nearby, meta_type)
        coarse_xyz, features = levels.pop()
        for layer in (self.fp5, self.fp4, self.fp3, self.fp2, self.fp1):
            fine_xyz, skip = levels.pop()
            features = layer(fine_xyz, coarse_xyz, skip, features)
            coarse_xyz = fine_xyz
        logits = self.afterprocess(features.transpose(1, 2)).transpose(1, 2)
        return F.log_softmax(logits, dim=-1)


def _stage2_inputs(xyz, constraints, n_metatype):
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or xyz.size(1) == 0:
        raise ValueError("CSTNet expects a nonempty XYZ tensor [B, N, 3]")
    if constraints is None or constraints.ndim != 3 or constraints.shape[:2] != xyz.shape[:2]:
        raise ValueError("CSTNet requires per-point constraints aligned with XYZ")
    if constraints.size(-1) != 10:
        raise ValueError("CSTNet requires 10 legacy channels: direction(3), nearby one-hot(2), primitive(5)")
    return xyz, constraints[..., :3], constraints[..., 3:5], constraints[..., 5:5 + n_metatype]


class CstNetClassifier(CrossAttention_Cls):
    def forward(self, xyz, constraints):
        return super().forward(*_stage2_inputs(xyz, constraints, self.n_metatype))


class CstNetSegmenter(CrossAttention_Seg):
    def forward(self, xyz, constraints, constraint_masks=None):
        return super().forward(*_stage2_inputs(xyz, constraints, self.n_metatype))


def add_cstnet_arguments(parser):
    parser.add_argument("--cstnet_n_metatype", type=int, choices=(4, 5), default=5,
                        help="Number of primitive channels in the original CSTNet weights")
    parser.add_argument("--cstnet_weights", metavar="PATH",
                        help="Start a new CSTNet run from an original classification or segmentation state_dict")


def load_cstnet_weights(model, path):
    """Load the original model-only .pth strictly, before optimizer initialization."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
