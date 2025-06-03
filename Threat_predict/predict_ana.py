import json
import pandas as pd
path = 'results/predic_information/ana_info_combined_threshold_0.5.json'
with open(path, 'r', encoding='utf-8') as f:
    data_dict = json.load(f)

layer_data = data_dict['by_layer']


df = pd.DataFrame.from_dict({k: v for k, v in layer_data.items() if k.startswith('layer')}, 
                           orient='index')

# 按Precision降序排序
df_sorted = df.sort_values('Precision', ascending=False)


print(df_sorted.head(10))  # 打印前10行
# 保存为CSV
#df_sorted.to_csv('results/predic_information/layer_metrics_sorted.csv', index_label='Layer')

# 提取层号整数
top_10_layers = df_sorted.head(10).index.str.extract('(\d+)')[0].astype(int).tolist()
top_15_layers = df_sorted.head(15).index.str.extract('(\d+)')[0].astype(int).tolist() 
top_20_layers = df_sorted.head(20).index.str.extract('(\d+)')[0].astype(int).tolist()

print("Precision最高的10层:", top_10_layers)
print("Precision最高的15层:", top_15_layers) 
print("Precision最高的20层:", top_20_layers)