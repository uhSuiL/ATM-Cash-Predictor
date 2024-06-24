from train_simple_gru import train_simple_gru

import pandas as pd


if __name__ == '__main__':
	train_data = pd.read_csv('./data/13series_time_stacked_cash/train.csv').drop(['日期'], axis=1).astype(float)
	train_data.columns = [int(col.split('_')[-1]) for col in train_data.columns]

	cluster_info = pd.read_csv('./data/cluster/fft_agg_cluster_4.csv')
	cluster_info['index'] = cluster_info['index'].apply(eval)

	for id_list in cluster_info['index']:
		clustered_train_data = train_data[id_list]
		train_simple_gru(
			train_data=clustered_train_data,
			save_dir=f'./log/ClusteredSimpleGRU/{tuple(id_list)}/',
			NUM_EPOCH=200,
			BATCH_SIZE=4,
		)

	print("complete")
