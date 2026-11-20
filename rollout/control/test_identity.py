import copy
import unittest

from common import check_handshake, manifest, verify_package


class IdentityTests(unittest.TestCase):
    def test_actual_package(self):
        self.assertEqual(verify_package(require_weights=False)['checkpoint_step'], 15490)

    def test_old_sft_handshake_is_rejected(self):
        identity = manifest()
        old = {key: value for key, value in identity.items() if not key.startswith('recap_')}
        with self.assertRaises(RuntimeError):
            check_handshake(old, identity)

    def test_wrong_norm_camera_or_guidance_is_rejected(self):
        identity = manifest()
        for key, value in [('normalization_sha256', 'old-norm'),
                           ('camera_layout', 'physical_right_wrist'),
                           ('recap_enabled', False), ('recap_conditioning', 'timestep'),
                           ('recap_guidance_scale', 1.5), ('zero_state', 'on')]:
            with self.subTest(key=key):
                metadata = copy.deepcopy(identity)
                metadata[key] = value
                with self.assertRaises(RuntimeError):
                    check_handshake(metadata, identity)

    def test_matching_recap_identity_is_accepted(self):
        check_handshake(manifest(), manifest())


if __name__ == '__main__':
    unittest.main()
