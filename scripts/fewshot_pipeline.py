"""Fixed audited training demonstrations; same encoding for normalized or raw rows."""
def prepare(tokenizer,rows,settings,training_sources=False):
 from fewshot_readout import encode
 demos=settings['_demonstrations'];result=[]
 for row in rows:
  if training_sources:
   question=row['request']['questions']['q'];state=row['request']['state']
  else:question=row['question'];state=row['state']
  encoding=encode(tokenizer,state,question,demos)
  item={'id':row['id'],'type':question['type'],'encoding':encoding}
  if training_sources:item.update(source=row['source'],target_index=encoding['option_ids'].index(row['targets']['q']))
  result.append(item)
 return result

def predict(model,tokenizer,prepared,settings,emit):
 from pretrained_readout import predict as native_predict
 return native_predict(model,tokenizer,prepared,batch_size=settings['batch_size'],emit=emit)
