"""Synthetic guards for the reconstruction amendment; no source/STAGES records."""
import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
import numpy as np
from reconstruct_challenge_normalization import ENTRY, META_SHA
from reconstruct_challenge_normalization import download_file, save_feature_taps, digest
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

    def test_feature_tap_cache_retains_exact_arrays_without_modifying_input(self):
        rng=np.random.default_rng(46)
        taps={name:rng.normal(size=shape).astype(np.float32) for name,shape in
              [('latent',(1024,)),('global',(2048,)),('temporal',(32,32)),('counterfactual',(8,))]}
        before={k:v.copy() for k,v in taps.items()}
        with tempfile.TemporaryDirectory() as folder:
            meta=save_feature_taps(Path(folder),'synthetic',taps)
            path=Path(folder)/'feature_taps'/meta['feature_tap_file']
            self.assertEqual(digest(path),meta['feature_tap_sha256'])
            with np.load(path) as saved:
                for k in taps:
                    np.testing.assert_array_equal(saved[k],before[k])
                    np.testing.assert_array_equal(taps[k],before[k])

    def test_invalid_feature_tap_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            taps={'latent':np.zeros(1024),'global':np.zeros(2048),'temporal':np.zeros((32,32)),'counterfactual':np.full(8,np.nan)}
            with self.assertRaises(ValueError):save_feature_taps(Path(folder),'synthetic',taps)


if __name__=='__main__':unittest.main()
