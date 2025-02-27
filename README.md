
# **Pratilipi Recommendation System**

## **Project Overview**
This project aims to build a recommendation system for **Pratilipi** using collaborative filtering and content-based features. The model is trained using the **LightFM** algorithm, which leverages both user-item interactions and item features to recommend relevant items.

## **Data Preparation**
The data consists of two CSV files:
1. **user_interaction.csv**: Contains user-item interaction data (e.g., users' engagement with pratilipis).
2. **metadata.csv**: Contains metadata about items (pratilipis), such as categories.

### **Steps in Data Preprocessing**:
1. **Loading Data**: The data is loaded from Google Drive using `pandas`.
   ```python
   user_interactions = pd.read_csv('/content/drive/My Drive/user_interaction.csv')
   meta_data = pd.read_csv('/content/drive/My Drive/metadata.csv')
   ```
   
2. **Time-based Sorting**: The `updated_at` column is converted to datetime, and interactions are sorted by the latest activity.
   ```python
   user_interactions['updated_at'] = pd.to_datetime(user_interactions['updated_at'])
   user_interactions = user_interactions.sort_values(by='updated_at')
   ```

3. **Data Integrity Check**: The notebook verifies that the data is loaded correctly and checks for missing values in both user interactions and metadata.

## **Model Training**
The recommendation model is built using the **LightFM** library, which combines collaborative filtering and content-based filtering. The model uses the **WARP loss function** (Weighted Approximate-Rank Pairwise), which optimizes the ranking of relevant items based on user-item interactions.

### **Training the Model**:
```python
model = LightFM(loss='warp', random_state=42)
model.fit(interactions, item_features=item_features, epochs=30, num_threads=4)
```

#### **What Happens During Model Fitting**:
- **Model Initialization**: The model is initialized with the **WARP loss function**, and a random state is set to ensure reproducibility. The model is then trained using the **user-item interactions** matrix and **item features** (such as categories).
  
- **Item Features**: The `item_features` matrix represents the content-based features of items (e.g., pratilipi categories). This helps the model incorporate item attributes into the learning process, beyond just user-item interactions.

- **Training**: The model is trained using the **user-item interaction matrix** (`interactions`) and **item features** (`item_features`) for **30 epochs**. The training process aims to optimize the model's ability to predict user-item interactions based on both historical data (collaborative filtering) and item characteristics (content-based filtering).
  
- **Epochs and Threads**: The model iterates over the data for **30 epochs** and uses **4 threads** to speed up the computation by parallelizing the task.

---

### **WARP Loss Function** (Weighted Approximate-Rank Pairwise)

The **WARP loss function** is a key component of the **LightFM** model, particularly designed for ranking tasks. It is one of the loss functions available in **LightFM** and is specifically suited for **ranking** problems where the goal is to optimize the order of items, rather than just predicting ratings.

#### **How WARP Works**:
- **Pairwise Ranking**: WARP focuses on **pairwise ranking** by considering a positive item (an item the user interacted with) and a negative item (an item the user did not interact with). The idea is to rank the positive item higher than the negative item.
  
- **Training with WARP**: During training, the model tries to push the score of the positive item higher and the score of the negative item lower, optimizing the embeddings of both the user and the item to achieve this.
  
- **Sampling**: The model doesn’t consider all negative items but instead samples a small number of **negative items** (not interacted with by the user) to update the embeddings. This makes WARP more efficient than other methods that consider all possible items.

- **Weighted Loss**: WARP gives **more weight** to examples where the model makes larger errors, improving the model’s ability to focus on hard-to-rank items. This can improve the overall ranking performance of the model.

#### **Why Use WARP?**
- **Efficient Ranking**: It is particularly effective for tasks where ranking is more important than predicting exact values (e.g., predicting which items should be recommended to users, rather than predicting ratings).
  
- **Better Performance on Sparse Data**: WARP performs well even when the user-item interaction matrix is **sparse**, which is common in recommendation systems where users interact with only a small subset of items.

- **Training with WARP** helps the model focus on **correct ranking**, ensuring that the relevant items are ranked higher for each user.

For more detailed information on the WARP loss function, refer to the original paper: [WARP Loss Paper - ArXiv](https://arxiv.org/pdf/1507.08439).

---

## **Evaluation**
The model is evaluated using three key metrics:
1. **Precision@5**: Measures how many of the top 5 recommended items are relevant to the user.
2. **Recall@5**: Measures how many relevant items appear in the top 5 recommendations.
3. **AUC**: Measures the model’s ability to rank relevant items higher than irrelevant ones.

### **Evaluation Results**:
- **Precision@5**: 0.0009
- **Recall@5**: 0.0173
- **AUC**: 0.8260

## **Discussion on Precision and Recall**
Even though the **AUC** is relatively high (0.8260), indicating that the model is good at distinguishing between relevant and irrelevant items, both **Precision@20** and **Recall@20** are quite low.

### **Possible Causes**:
1. **High Number of Items**: With a large number of pratilipis, the model might struggle to recommend highly relevant items in the top 5 recommendations, leading to low precision and recall.
2. **Cold Start Problem**: If there are users in the test set with no interactions in the training data, the model will struggle to provide relevant recommendations, especially for these cold start users.
3. **Model Tuning**: The low precision and recall could be due to insufficient training or suboptimal hyperparameters (e.g., learning rate, number of epochs, regularization).

### **Next Steps for Improvement**:
1. **Hyperparameter Tuning**: Experiment with different learning rates, number of epochs, and regularization parameters.
2. **Data Augmentation**: Add more user-item interactions or improve feature engineering to enhance the model's ability to make diverse and relevant recommendations.
3. **Cold Start Handling**: Incorporate more user features (such as demographics) to improve recommendations for cold start users.
4. **Hybrid Model**: Combine collaborative filtering with more content-based methods to address low precision and recall.

## **References**
1. **LightFM GitHub Repository**: [https://github.com/lyst/lightfm](https://github.com/lyst/lightfm)
2. **WARP Loss Paper (ArXiv)**: [https://arxiv.org/pdf/1507.08439](https://arxiv.org/pdf/1507.08439)

