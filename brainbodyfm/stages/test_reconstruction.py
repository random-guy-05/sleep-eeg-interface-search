"""Synthetic guards for the reconstruction amendment; no source/STAGES records."""
import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
import numpy as np
from reconstruct_challenge_normalization import ENTRY, META_SHA
from reconstruct_challenge_normalization import download_file
from run_reconstructed_rank import validate_reconstruction
from age_conditioned_transfer import STABLE4, fixed_score


class ReconstructionGuards(unittest.TestCase):
    def fixture(self):
        return dict(reconstructed=True,original_organizer_artifact=False,entry_commit=ENTRY,
                    dataset_version=2,metadata_sha256=META_SHA,
                    normalizer=dict(columns=list(STABLE4),direction=[1.,1.,-1.,-1.],
                                    weights=[.36,.27,.235,.135,0.],feat_mean=[0.]*5,feat_scale=[1.]*5))

    def test_reconstruction_cannot_claim_original(self):
        x=self.fixture();x['original_organizer_artifact']=True
        with self.assertRaises(ValueError):validate_reconstruction(x)

    def test_wrong_release_and_sign_are_rejected(self):
        x=self.fixture();x['dataset_version']=1
        with self.assertRaises(ValueError):validate_reconstruction(x)
        x=self.fixture();x['normalizer']['direction'][-1]=1.
        with self.assertRaises(AssertionError):validate_reconstruction(x)

    def test_source_normalizer_is_fixed_for_target_rows(self):
        n=validate_reconstruction(self.fixture());rng=np.random.default_rng(7);x=rng.normal(size=(20,4))
        original=fixed_score(x,n);x[-1]*=1e7
        np.testing.assert_array_equal(original[:-1],fixed_score(x,n)[:-1])

    def test_means_do_not_change_pairwise_ranking(self):
        x=self.fixture();n=validate_reconstruction(x);rng=np.random.default_rng(8);heads=rng.normal(size=(40,4))
        first=fixed_score(heads,n);n['feat_mean']=[2.,8.,-3.,1.,0.]
        second=fixed_score(heads,n)
        np.testing.assert_array_equal(np.argsort(first),np.argsort(second))

    def test_download_request_pins_large_version_two_and_preserves_nested_path(self):
        def fake_run(command, check):
            import json
            request=json.loads(command[command.index('--data-binary')+1])
            self.assertEqual(request['datasetVersionNumber'],2)
            self.assertEqual(request['fileName'],'physiological_data/site/synthetic.edf')
            self.assertEqual(request['datasetSlug'],'physionetchallenge2026datalargeversion')
            Path(command[command.index('-o')+1]).write_bytes(b'0       synthetic EDF')
        with tempfile.TemporaryDirectory() as folder, patch('reconstruct_challenge_normalization.subprocess.run',side_effect=fake_run):
            out=Path(folder)/'synthetic.edf'
            download_file('physiological_data/site/synthetic.edf',out)
            self.assertTrue(out.read_bytes().startswith(b'0       '))


if __name__=='__main__':unittest.main()
