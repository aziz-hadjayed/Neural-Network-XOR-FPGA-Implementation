import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
#----------------------------------------------------------------------------------data
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)
y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)
#----------------------------------------------------------------------------------construction
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation= 'sigmoid', input_shape=(2,)), # deux layers de deux entré 
    tf.keras.layers.Dense(1, activation = 'sigmoid' ) # neuron sortie
])
#----------------------------------------------------------------------------------summary
print("Résumé du modèle :")
model.summary()
#----------------------------------------------------------------------------------compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.1), # sgd
    loss='binary_crossentropy',
    metrics=['accuracy']
)
#----------------------------------------------------------------------------------trainning
history = model.fit(
    X, y,
    epochs=150,
    verbose=1,
    validation_data=(X, y)       
)
#----------------------------------------------------------------------------------affichage courbes
# 5. Affichage des courbes loss / accuracy
plt.figure()
plt.plot(history.history['loss'], label='Loss entraînement')
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='Loss validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Évolution de la loss')
plt.legend()
plt.grid(True)
plt.show()
plt.figure()
plt.plot(history.history['accuracy'], label='Accuracy entraînement')
if 'val_accuracy' in history.history:
    plt.plot(history.history['val_accuracy'], label='Accuracy validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Évolution de l’accuracy')
plt.legend()
plt.grid(True)
plt.show()
#----------------------------------------------------------------------------------prediction
# 6. Prédictions du modèle
preds = model.predict(X)
print("\nPrédictions sur la table XOR :")
for i, (x, y_true, y_pred) in enumerate(zip(X, y, preds)):
    print(f"Entrée: {x}  →  Vrai: {int(y_true[0])}  |  Prédit: {y_pred[0]:.4f}  |  Arrondi: {int(y_pred[0] > 0.5)}")
#----------------------------------------------------------------------------------affichage W and bias
# 7. Affichage des poids et du biais
print("\nPoids et biais appris :")
for layer in model.layers:
    weights, bias = layer.get_weights()
    print("Poids (w) :", weights.reshape(-1))
    print("Biais (b) :", bias)
print("\n=== Caractéristiques des couches ===")
for i, layer in enumerate(model.layers):
    print(f"\n----- Layer {i} -----")
    print("Nom        :", layer.name)
    print("Type       :", layer.__class__.__name__)
    # Certaines couches n'ont pas d'activation (ex: InputLayer)
    if hasattr(layer, 'activation'):
        print("Activation :", layer.activation.__name__)
    # Poids et biais (si la couche en a)
    weights = layer.get_weights()
    if weights:
        W, b = weights
        print("Poids W shape :", W.shape)
        print(W)
        print("Biais b shape :", b.shape)
        print(b)
    else:
        print("Pas de poids / biais (couche sans paramètres)")
# ---------------------------------------------------------------------------------- frontière de décision
def plot_decision_boundary(model, X, y):
    """
    Affiche la frontière de décision apprise par le modèle.
    """
    # Définir l'étendue du graphique
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    # Créer une grille de points (mesh)
    h = 0.01  # Taille du pas dans la grille
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Prédire la sortie pour tous les points de la grille
    # On aplatit les coordonnées de la grille pour les passer au modèle
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points, verbose=0)
    
    # Z est la probabilité de la classe 1. On l'arrondit à 0 ou 1 pour la classification
    Z = Z.reshape(xx.shape)
    
    # Afficher la frontière de décision en utilisant un contour rempli
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu, levels=1) # levels=1 sépare à Z=0.5
    
    # Afficher les points d'entraînement
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap=plt.cm.RdYlBu, edgecolors='k', marker='o', s=100)
    
    # Ajouter les labels et le titre
    plt.title('Frontière de Décision pour XOR (Activation sigmoid)')
    plt.xlabel('Entrée 1')
    plt.ylabel('Entrée 2')
    
    # Afficher les coordonnées spécifiques du XOR pour la clarté
    for i, (x, target) in enumerate(zip(X, y)):
        plt.annotate(f'({x[0]}, {x[1]})', (x[0], x[1]), textcoords="offset points", xytext=(5,5), ha='center')

    plt.show()

# Appeler la fonction pour afficher la frontière de décision
plot_decision_boundary(model, X, y)
