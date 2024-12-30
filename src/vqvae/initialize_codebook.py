# from sklearn.cluster import KMeans

# # Assume `all_embeddings` is a tensor of shape [num_samples, emb_dim] collected from your PartialVAE
# all_embeddings_np = all_embeddings.numpy()  # Convert to NumPy array for sklearn
# kmeans = KMeans(n_clusters=codebook_size, random_state=0).fit(all_embeddings_np)

# initial_codebook = torch.tensor(kmeans.cluster_centers_, dtype=torch.float)
