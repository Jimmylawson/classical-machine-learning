import numpy as np


class KMeans:
    def __init__(self,k=5,max_iteration=300,tolerance=1e-4,random_state=42):
        self.k = k
        self.max_iteration = max_iteration
        self.tolerance = tolerance
        self.random_state = random_state

        self.centroids = None
        self.labels = None
        self.inertia = None
        self.n_iterations = None


    def initialize_centroids(self, X):
        # Select K starting customer points
        rng = np.random.default_rng(self.random_state)
        centroid_indexes = rng.choice(
            X.shape[0], # number of customers
            size=self.k,
            replace=False #prevents the same customer from being selected more than once.
        )
        self.centroids =  X[centroid_indexes].copy()#retrieve those five customer row from X


        return self.centroids



    def calculate_distances(self, X):
        # Calculate every customer's distance
        # from every centroid
        difference  = X[:, np.newaxis,:] - self.centroids[np.newaxis, :,:]
        squared_distance = np.sum(difference ** 2,axis=2)

        return squared_distance


    def assign_clusters(self, X):
        # Assign every customer to the closest centroid
        squared_distances = self.calculate_distances(X)
        labels = np.argmin(squared_distances, axis=1)
        return labels





    def update_centroids(self, X, labels):
        # Calculate the mean position of each cluster
        new_centroids = np.zeros(
            (self.k, X.shape[1]),
            dtype=float,
        )

        for cluster_index in range(self.k):
            cluster_customers = X[
                labels == cluster_index
            ]
            if len(cluster_customers) == 0:
                new_centroids[cluster_index] = (
                    self.centroids[cluster_index]
                )
            else:
                new_centroids[cluster_index] = (
                    cluster_customers.mean(axis=0)
                )

        return new_centroids



    def calculate_inertia(self, X,labels):
        # Calculate the total squared distance
        assigned_centroid = self.centroids[labels]
        squared_differences = (X - assigned_centroid) ** 2

        inertia = np.sum(squared_differences)
        return inertia


    def fit(self, X):
        #Coordinate the full training loop
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")
        if self.k <= 0:
            raise ValueError("k must be greater than zero.")
        if self.k > X.shape[0]:
            raise ValueError(
                "k cannot exceed the number of customers."
            )
        if self.max_iteration <= 0:
            raise ValueError(
                "max_iteration must be greater than zero."
            )

        self.initialize_centroids(X)

        for iteration in range(self.max_iteration):
            labels = self.assign_clusters(X)
            new_centroids = self.update_centroids(X, labels)

            centroid_movement = np.linalg.norm(
                new_centroids - self.centroids
            )

            self.centroids = new_centroids

            if centroid_movement < self.tolerance:
                break

        self.labels = self.assign_clusters(X)
        self.inertia = float(
            self.calculate_inertia(X, self.labels)
        )
        self.n_iterations = iteration + 1

        return self


    def predict(self, X):
        # Assign new customers to learned centroids
        if self.centroids is None:
            raise ValueError(
                "Model has not been fitted. Call fit() first."
            )

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")
        if X.shape[1] != self.centroids.shape[1]:
            raise ValueError(
                f"Expected {self.centroids.shape[1]} features, "
                f"but received {X.shape[1]}."
            )

        return self.assign_clusters(X)
