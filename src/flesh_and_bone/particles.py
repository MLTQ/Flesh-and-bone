"""Fixed-capacity mobile Gaussian-cell reservoir and differentiation state."""

import torch

from .morphology import deform_embedded, embed_points


class ParticleSystem:
    """Mutable H0 particle reservoir with explicit niche reservations."""

    def __init__(self, capacity, recurrent_channels=16, bone_count=5,
                 device=None, dtype=torch.float32):
        self.capacity = int(capacity)
        self.bone_count = int(bone_count)
        if self.bone_count < 1:
            raise ValueError("bone_count must be positive")
        self.positions = torch.zeros(self.capacity, 3, device=device, dtype=dtype)
        self.velocities = torch.zeros_like(self.positions)
        self.mass = torch.zeros(self.capacity, device=device, dtype=dtype)
        self.active = torch.zeros(self.capacity, device=device, dtype=torch.bool)
        self.assigned_site = torch.full(
            (self.capacity,), -1, device=device, dtype=torch.long
        )
        self.committed = torch.zeros_like(self.active)
        self.material_locked = torch.zeros_like(self.active)
        self.generation = torch.full_like(self.assigned_site, -1)
        self.guide_region = torch.full_like(self.assigned_site, -1)
        self.maturation_steps = torch.zeros_like(self.assigned_site)
        self.part = torch.full_like(self.assigned_site, -1)
        self.checker = torch.full_like(self.assigned_site, -1)
        self.splat_scale = torch.ones(
            self.capacity, device=device, dtype=dtype
        )
        self.recurrent = torch.zeros(
            self.capacity, recurrent_channels, device=device, dtype=dtype
        )
        self.bone_weights = torch.zeros(
            self.capacity, self.bone_count, device=device, dtype=dtype
        )
        self.bone_local = torch.zeros(
            self.capacity, self.bone_count, 3, device=device, dtype=dtype
        )

    @staticmethod
    def _splat_scale_at(body_plan, sites):
        if hasattr(body_plan, "splat_scale"):
            return body_plan.splat_scale[sites]
        return body_plan.reference_sites.new_ones(sites.shape)

    @property
    def device(self):
        return self.positions.device

    @property
    def dtype(self):
        return self.positions.dtype

    @property
    def active_count(self):
        return int(self.active.sum().item())

    @property
    def committed_count(self):
        return int(self.committed.sum().item())

    def feed(self, body_plan, target_positions, count, generator,
             source=(0.0, -1.35, 0.0), jitter=0.055):
        """Activate cells and reserve nearest currently unfilled niches."""
        inactive = torch.nonzero(~self.active).flatten()
        reserved = torch.zeros(
            body_plan.site_count, device=self.device, dtype=torch.bool
        )
        assigned = self.assigned_site[self.active]
        assigned = assigned[assigned >= 0]
        if assigned.numel():
            reserved[assigned] = True
        available_sites = torch.nonzero(~reserved).flatten()
        actual = min(int(count), inactive.numel(), available_sites.numel())
        if actual <= 0:
            return 0

        source_tensor = self.positions.new_tensor(source)
        site_distance = (target_positions[available_sites] - source_tensor).norm(dim=1)
        selected_sites = available_sites[site_distance.argsort()[:actual]]
        selected_particles = inactive[:actual]
        noise = torch.randn(
            actual, 3, device=self.device, dtype=self.dtype, generator=generator
        ) * jitter
        self.positions[selected_particles] = source_tensor + noise
        self.velocities[selected_particles] = 0
        self.mass[selected_particles] = 1
        self.active[selected_particles] = True
        self.assigned_site[selected_particles] = selected_sites
        self.committed[selected_particles] = False
        self.material_locked[selected_particles] = False
        self.generation[selected_particles] = 0
        self.guide_region[selected_particles] = -1
        self.maturation_steps[selected_particles] = 0
        self.part[selected_particles] = -1
        self.checker[selected_particles] = -1
        self.splat_scale[selected_particles] = self._splat_scale_at(
            body_plan, selected_sites
        )
        self.recurrent[selected_particles] = 0
        self.bone_weights[selected_particles] = 0
        self.bone_local[selected_particles] = 0
        return actual

    def feed_unassigned(self, count, generator, source=(0.0, -1.35, 0.0),
                        jitter=0.055, generation=0):
        """Activate cells without exposing any target-site identity."""
        inactive = torch.nonzero(~self.active).flatten()
        actual = min(int(count), inactive.numel())
        if actual <= 0:
            return 0
        selected = inactive[:actual]
        source_tensor = self.positions.new_tensor(source)
        noise = torch.randn(
            actual, 3, device=self.device, dtype=self.dtype, generator=generator
        ) * jitter
        self.positions[selected] = source_tensor + noise
        self.velocities[selected] = 0
        self.mass[selected] = 1
        self.active[selected] = True
        self.assigned_site[selected] = -1
        self.committed[selected] = False
        self.material_locked[selected] = False
        self.generation[selected] = int(generation)
        self.guide_region[selected] = -1
        self.maturation_steps[selected] = 0
        self.part[selected] = -1
        self.checker[selected] = -1
        self.splat_scale[selected] = 1
        self.recurrent[selected] = 0
        self.bone_weights[selected] = 0
        self.bone_local[selected] = 0
        return actual

    def update_commitment(self, body_plan, target_positions, threshold):
        """Persistently differentiate cells that reach their reserved niche."""
        candidates = self.active & ~self.committed & (self.assigned_site >= 0)
        indices = torch.nonzero(candidates).flatten()
        if not indices.numel():
            return 0
        sites = self.assigned_site[indices]
        distance = (self.positions[indices] - target_positions[sites]).norm(dim=1)
        selected = indices[distance <= threshold]
        if not selected.numel():
            return 0
        selected_sites = self.assigned_site[selected]
        self.committed[selected] = True
        self.material_locked[selected] = True
        self.part[selected] = body_plan.dominant_bone[selected_sites]
        self.checker[selected] = body_plan.checker[selected_sites]
        self.splat_scale[selected] = self._splat_scale_at(
            body_plan, selected_sites
        )
        self.bone_weights[selected] = body_plan.bone_weights[selected_sites]
        self.bone_local[selected] = body_plan.bone_local[selected_sites]
        return int(selected.numel())

    def update_continuous_commitment(self, body_plan, frame, target_positions,
                                     threshold):
        """Differentiate cells using only proximity to the continuous body field."""
        candidates = self.active & ~self.committed
        indices = torch.nonzero(candidates).flatten()
        if not indices.numel():
            return 0
        distance = torch.cdist(self.positions[indices], target_positions)
        nearest_distance, nearest_site = distance.min(dim=1)
        selected_mask = nearest_distance <= threshold
        selected = indices[selected_mask]
        if not selected.numel():
            return 0
        selected_sites = nearest_site[selected_mask]
        weights, local = embed_points(
            self.positions[selected], frame, body_plan.weight_sigma
        )
        self.committed[selected] = True
        self.material_locked[selected] = False
        self.maturation_steps[selected] = 0
        self.part[selected] = weights.argmax(dim=1)
        self.checker[selected] = body_plan.checker[selected_sites]
        self.splat_scale[selected] = self._splat_scale_at(
            body_plan, selected_sites
        )
        self.bone_weights[selected] = weights
        self.bone_local[selected] = local
        return int(selected.numel())

    def refresh_continuous_attachments(self, body_plan, frame):
        """Track migrating committed material in bone coordinates before motion."""
        selected = torch.nonzero(
            self.active & self.committed & ~self.material_locked
        ).flatten()
        if not selected.numel():
            return
        weights, local = embed_points(
            self.positions[selected], frame, body_plan.weight_sigma
        )
        self.bone_weights[selected] = weights
        self.bone_local[selected] = local
        self.part[selected] = weights.argmax(dim=1)

    def refresh_plastic_material(self, body_plan, target_positions):
        """Read phenotype from the nearest local material-field sample."""
        selected = torch.nonzero(
            self.active & self.committed & ~self.material_locked
        ).flatten()
        if not selected.numel():
            return
        nearest = torch.cdist(
            self.positions[selected], target_positions
        ).argmin(dim=1)
        self.checker[selected] = body_plan.checker[nearest]
        self.splat_scale[selected] = self._splat_scale_at(body_plan, nearest)

    def lock_material(self):
        """End developmental plasticity without changing current phenotype."""
        selected = self.active & self.committed
        newly_locked = selected & ~self.material_locked
        self.material_locked[selected] = True
        return int(newly_locked.sum().item())

    def update_local_maturation(self, indices, stable, required_steps):
        """Lock cells after consecutive locally stable developmental steps."""
        eligible = self.committed[indices] & ~self.material_locked[indices]
        progressing = eligible & stable
        current = self.maturation_steps[indices]
        current = torch.where(progressing, current + 1, torch.zeros_like(current))
        self.maturation_steps[indices] = current
        matured = eligible & (current >= int(required_steps))
        selected = indices[matured]
        self.material_locked[selected] = True
        return int(selected.numel())

    def remove(self, indices):
        """Deactivate an explicit set of cells and return their mass to capacity."""
        indices = torch.as_tensor(indices, device=self.device, dtype=torch.long)
        indices = indices[self.active[indices]]
        if not indices.numel():
            return 0
        self.active[indices] = False
        self.mass[indices] = 0
        self.velocities[indices] = 0
        self.assigned_site[indices] = -1
        self.committed[indices] = False
        self.material_locked[indices] = False
        self.generation[indices] = -1
        self.guide_region[indices] = -1
        self.maturation_steps[indices] = 0
        self.part[indices] = -1
        self.checker[indices] = -1
        self.splat_scale[indices] = 1
        self.recurrent[indices] = 0
        self.bone_weights[indices] = 0
        self.bone_local[indices] = 0
        return int(indices.numel())

    def attachment_targets(self, frame, indices=None):
        """Deform persistent material coordinates for active/selected particles."""
        if indices is None:
            indices = torch.nonzero(self.active).flatten()
        return deform_embedded(
            self.bone_weights[indices], self.bone_local[indices], frame
        )

    def active_tensors(self):
        """Return active indices and their state for dynamics/metrics."""
        indices = torch.nonzero(self.active).flatten()
        return (
            indices,
            self.positions[indices],
            self.velocities[indices],
            self.assigned_site[indices],
        )
