"""Native MuJoCo model identity checks; no GPU or training claim."""

import copy
import unittest

import mujoco
import numpy as np

from hexapod_mjlab.common.checkpoint import (
    _numeric_signature, compare_model_physics, model_physics_identity,
)


def model():
    return mujoco.MjModel.from_xml_string('''
      <mujoco><worldbody><geom type="plane" size="10 10 .1"/>
        <body pos="0 0 1"><freejoint/>
          <geom type="box" size=".1 .2 .3" mass="10"/>
          <body><joint name="hinge" range="-90 90" damping="1"/>
            <geom type="sphere" size=".1" mass="1"/></body>
        </body></worldbody><actuator><motor joint="hinge"/></actuator>
      </mujoco>''')


class PhysicalIdentityTests(unittest.TestCase):
    def test_native_models_and_visual_changes_are_compatible(self):
        a, b = model(), model()
        b.geom_rgba[:] = .3
        compare_model_physics(model_physics_identity(a), model_physics_identity(b))

    def test_dynamics_changes_are_rejected_by_field(self):
        changes = {
            "body_mass": lambda m: m.body_mass.__setitem__(1, 20),
            "body_inertia": lambda m: m.body_inertia.__setitem__((1, 0), .5),
            "dof_damping": lambda m: m.dof_damping.__setitem__(-1, 2),
            "jnt_range": lambda m: m.jnt_range.__setitem__((1, 1), 2),
            "geom_friction": lambda m: m.geom_friction.__setitem__((0, 0), .2),
            "geom_solref": lambda m: m.geom_solref.__setitem__((0, 0), .03),
            "geom_size": lambda m: m.geom_size.__setitem__((1, 0), .2),
            "actuator_gainprm": lambda m: m.actuator_gainprm.__setitem__((0, 0), 2),
            "opt.gravity": lambda m: m.opt.gravity.__setitem__(2, -5),
        }
        saved = model_physics_identity(model())
        for field, change in changes.items():
            with self.subTest(field=field):
                changed = model()
                change(changed)
                with self.assertRaisesRegex(ValueError, field):
                    compare_model_physics(model_physics_identity(changed), saved)

    def test_missing_old_unknown_or_incomplete_signatures_fail_closed(self):
        current = model_physics_identity(model())
        for saved in (None, {}, {"schema": "unknown", "fields": {}}):
            with self.subTest(saved=saved), self.assertRaisesRegex(ValueError, "weights-only"):
                compare_model_physics(current, saved)
        saved = copy.deepcopy(current)
        saved["fields"].pop("body_mass")
        with self.assertRaisesRegex(ValueError, "field set"):
            compare_model_physics(current, saved)

    def test_numeric_signature_ignores_byte_order_and_signed_zero(self):
        self.assertEqual(_numeric_signature(np.array([0., 1.], dtype=">f8")),
                         _numeric_signature(np.array([-0., 1.], dtype="<f8")))
        with self.assertRaisesRegex(ValueError, "NaN"):
            _numeric_signature(np.array([float("nan")]))


if __name__ == "__main__":
    unittest.main()
