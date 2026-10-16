import json,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock
from durable_modal_calls import invoke

class DurableCalls(unittest.TestCase):
 def test_handle_saved_before_wait_and_recovered_without_spawn(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'call.json';request={'experiment':'fixed'}
   def result():
    self.assertEqual(json.loads(path.read_text())['function_call_id'],'fc-saved')
    raise TimeoutError('Host observation interrupted')
   call=Mock(object_id='fc-saved');call.get.side_effect=result;spawn=Mock(return_value=call)
   with self.assertRaises(TimeoutError):invoke(path,request,spawn,Mock())
   restored=Mock();restored.get.return_value={'status':'complete'};restore=Mock(return_value=restored)
   self.assertEqual(invoke(path,request,spawn,restore),{'status':'complete'})
   spawn.assert_called_once();restore.assert_called_once_with('fc-saved')
 def test_foreign_request_rejected_without_submitting(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'call.json';path.write_text(json.dumps({'request':{'id':1},'function_call_id':'old'}));spawn=Mock();restore=Mock()
   with self.assertRaises(ValueError):invoke(path,{'id':2},spawn,restore)
   spawn.assert_not_called();restore.assert_not_called()
if __name__=='__main__':unittest.main()
