import json
import os

from tqdm import tqdm
from openai import OpenAI
client = OpenAI(api_key='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')

dir = 'semeval_data'
json_path = os.listdir(dir)

for path in json_path:
  target_path = os.path.join(dir,path)
  print(f'# Processing {path}')
  with open(target_path,'r',encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]

  record = []
  for d in tqdm(dataset):
    messages = d['messages']
    query = messages[1]['content']
    answer = messages[2]['content']

    response = client.chat.completions.create(
      model="ft:gpt-3.5-turbo-1106:2023python::8RyWR8GA",
      messages=[
        {"role": "system", "content": "I am a factual chatbot that is excellent in word sense disambiguation."},
        {"role": "user", "content": query}
      ]
    )
    output = response.choices[0].message.content
    record.append(
      {
        'query': query,
        'gold_answer': answer,
        'gpt_output': output
      }
    )

  name = path.split('_')[0] + '_result.json'
  outfile_path = os.path.join('semeval_gpt_result',name)
  with open(outfile_path,'w',encoding='utf-8') as f:
    json.dump(record,f,indent=3,ensure_ascii=False)