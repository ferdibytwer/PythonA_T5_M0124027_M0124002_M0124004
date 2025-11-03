import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler as SS

# Read CSV
df = pd.read_csv('hambatan-siswa.csv')

# Scaling Data (Normalisasi)
df_numerik = df.drop(columns=['Nama Kelas', 'No']) # hilangkan kolom nama kelas dan no
scaler = SS()
scaled_data = scaler.fit_transform(df_numerik)
print(df_numerik)

# Kmeans Clustering
custom_centroids = np.array([[13,1,8], [27, 6, 40]])
kmeans = KMeans(n_clusters=2, init=custom_centroids, n_init=1, random_state=0 )
df['Cluster'] = kmeans.fit_predict(df_numerik)

# Interpretasi Cluster
conditions = [
    (df['Cluster']==0),
    (df['Cluster']==1)]
choices = ['Rendah', 'Tinggi']
df['Kategori Hambatan'] = np.select(conditions, choices, default='Tidak Diketahui')

# Tampilkan kelas dengan kategori hambatan pembelajaran online TINGGI
# print(df[['Nama Kelas', 'Cluster', 'Kategori Hambatan']])
print(df[df['Kategori Hambatan'] == 'Tinggi'][['Nama Kelas', 'Cluster', 'Kategori Hambatan']])
print('Jumlah iterasi :', kmeans.n_iter_)

