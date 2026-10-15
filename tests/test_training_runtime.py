"""Regression coverage for observed Modal launch and export failures."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from training_data import mounted_path
from export_training import export

class RuntimeTests(unittest.TestCase):
    def test_mount_symlink_and_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); actual=root/'actual';actual.mkdir()
            mount=root/'mount';mount.symlink_to(actual, target_is_directory=True)
            self.assertEqual(mounted_path(mount,'manifest.json'), actual.resolve()/'manifest.json')
            with self.assertRaises(ValueError): mounted_path(mount,'../outside')
            (actual/'escape').symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError): mounted_path(mount,'escape/outside')

    def test_actual_qwen_hybrid_bf16_forward_and_head_backward(self):
        import torch
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
        from vendor.clef_joint_schema import JointSchemaHead, collate_records
        from train_joint import forward_frozen, prepare
        from test_training import fixtures, CharacterTokenizer
        from training_data import load_splits
        torch.set_num_threads(1)
        cfg=Qwen3_5TextConfig(vocab_size=129,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
            linear_key_head_dim=8,linear_value_head_dim=8,linear_num_key_heads=2,
            linear_num_value_heads=4,layer_types=['linear_attention','full_attention'])
        backbone=Qwen3_5ForCausalLM(cfg).to(torch.bfloat16).requires_grad_(False)
        head=JointSchemaHead(hidden_size=32,width=8,routing_layers=1,layers=1,heads=2,feedforward=16)
        with tempfile.TemporaryDirectory() as tmp:
            manifest,_=fixtures(Path(tmp)); splits,_=load_splits(manifest)
            _,encoded=prepare(splits['train'],CharacterTokenizer(),8192)[0]
            batch=collate_records([encoded],0,torch.device('cpu'))
            with torch.autocast('cpu',dtype=torch.bfloat16):
                outputs=forward_frozen(backbone,head,batch)
                loss=sum(x.float().logsumexp(0) for x in outputs[0])
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(any(p.grad is not None for p in head.parameters()))
            self.assertTrue(all(p.grad is None for p in backbone.parameters()))

    def test_export_explicit_files_and_reject_bad_evidence(self):
        payload=b'checkpoint'
        for bad in [None,'hash','path']:
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as tmp:
                dest=Path(tmp)/'export'
                name='../outside' if bad=='path' else 'nested/head.bin'
                expected='0'*64 if bad=='hash' else hashlib.sha256(payload).hexdigest()
                inventory={name:expected}
                def download(remote,local):
                    Path(local).write_bytes(json.dumps(inventory).encode() if remote.endswith('checksums.json') else payload)
                with patch('export_training.download_file',side_effect=download):
                    if bad:
                        with self.assertRaises(ValueError): export('run-id',dest)
                        self.assertFalse(dest.exists())
                    else:
                        export('run-id',dest)
                        self.assertEqual((dest/name).read_bytes(),payload)
                        with self.assertRaises(FileExistsError): export('run-id',dest)

class NestedExportTests(unittest.TestCase):
    def test_nested_completed_training_export_uses_exact_prefix(self):
        payload=b'locked checkpoint';sha=hashlib.sha256(payload).hexdigest();calls=[]
        with tempfile.TemporaryDirectory() as tmp:
            def download(remote,local):
                calls.append(remote);Path(local).write_bytes(json.dumps({'best-head.bin':sha}).encode() if remote.endswith('checksums.json') else payload)
            with patch('export_training.download_file',side_effect=download):
                export('run-id',Path(tmp)/'output','attempts/immutable-id/training')
            self.assertEqual(calls,['run-id/attempts/immutable-id/training/checksums.json','run-id/attempts/immutable-id/training/best-head.bin'])
    def test_subdirectory_escape_is_rejected_before_network(self):
        with tempfile.TemporaryDirectory() as tmp,patch('export_training.download_file') as download:
            for value in ['../other','attempts/../../other','/absolute','',r'attempts\..\other']:
                with self.subTest(value=value),self.assertRaises(ValueError):export('run-id',Path(tmp)/'output',value)
            download.assert_not_called()

if __name__=='__main__': unittest.main()
