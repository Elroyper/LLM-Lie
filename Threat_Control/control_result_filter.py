import json
import re
import pandas as pd



path = "results/steering_results/control/template_c_control_interval.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)


dataset_path = 'data/control_sample_dataset/test_control_df_all_types_60.csv'
dataset = pd.read_csv(dataset_path, encoding='utf-8')
responses = data['responses']
labels = data['ground_truth']


liars = []

def cot_extract(response):
    import re
    match = re.match(r"(.*?</think>)(.*)", response, re.DOTALL)
    
    if match:
        CoT = match.group(1)  # 提取第一部分，包括 </think>
        output = match.group(2)  # 提取第二部分
        return (CoT, output)
    else:
        return None
    
cots = [cot_extract(response)[0] if isinstance(cot_extract(response),tuple) else None for response in responses ]
direct_responses = [cot_extract(response)[1] if isinstance(cot_extract(response),tuple) else None for response in responses ]

for i, (response, label) in enumerate(zip(direct_responses, labels)):
        if response is not None:
            # 提取Yes/No响应
            response_clean = response.strip().lower()
            if response_clean.startswith('[response]'):
                response_clean = response_clean[10:].strip().lower().rstrip('.,!?')
        else:
            response_clean = None

        #print(f"response_clean: {response_clean}")
        #print(f"label: {label}")
        if response_clean is not None:
            response_clean = response_clean + "  "
            is_yes = response_clean[:3] == 'yes'
            is_no = response_clean[:2] == 'no'
            if is_yes:
                response_clean = 'yes'
            elif is_no:
                response_clean = 'no'
            else:
                response_clean = None
        
        if response_clean in ['yes', 'no']:
            if(response_clean == 'no' and label == 1) or(response_clean == 'yes' and label == 0) :
                liar_dict = {
                    'statement':dataset['statement'][i],
                    'label': label,
                    'question': dataset['question'][i],
                    'negative':dataset['negative'][i],
                    'cot': cots[i],
                    'response': response,
                }
                liars.append(liar_dict)

print(f"liars: {len(liars)}")
#with open('results/steering_results/control/liars.json', 'w', encoding='utf-8') as f:
#    json.dump(liars, f, ensure_ascii=False, indent=4)
liar_df = pd.DataFrame(liars)
#liar_df.to_csv('results/steering_results/control/liars.csv', index=False, encoding='utf-8')
cot_len = [len(cot) for cot in liar_df['cot']]
#print(f"cot_len: {cot_len}")
#cot_len最短的五个对应的liar_df数据
cot_len_df = liar_df.copy()
cot_len_df['cot_len'] = cot_len
cot_len_df = cot_len_df.sort_values(by='cot_len', ascending=True)
cot_len_df = cot_len_df.head(5)
print(cot_len_df)

