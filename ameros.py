import pandas as pd
import nltk
import numpy as np
import matplotlib.pyplot as plt
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV



def custom_error(y_true, y_pred):
    # Υπολογισμός απόκλισης από το κοντινότερο άκρο
    errors = np.zeros_like(y_true)
    for i in range(len(y_true)):
        min_error = abs(y_pred[i][0] - y_true.iloc[i]['date_min'])
        max_error = abs(y_pred[i][1] - y_true.iloc[i]['date_max'])
        errors[i] = min(min_error, max_error)
    return errors

# Διαβάζουμε τα δεδομένα από το αρχείο CSV, ορίζοντας τη μηχανή ως "python" 
data = pd.read_csv("iphi2802 (3).csv", sep="	", engine="python", header=0, encoding="utf-8")



# Ορίζουμε μια λίστα για να αποθηκεύσουμε τα tokens των επιγραφών
tokens_list = []

for text in data['text']:
    if isinstance(text, str):  # Ελέγχει αν η τιμή είναι string
        tokens = word_tokenize(text.lower())  # tokenization και μετατροπή σε πεζούς χαρακτήρες
        filtered_tokens = [token for token in tokens if token.isalpha()]  # Φιλτράρει τα tokens που περιέχουν μόνο γράμματα
        tokens_list.append(filtered_tokens)
        
    else:
        
        pass

'''
# Εκτύπωση της λίστας των tokens
for i, filtered_tokens in enumerate(tokens_list):
    print("Tokens of inscription", i+1, ":", filtered_tokens)
'''

# Δημιουργία ενός TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Χρησιμοποιήστε μόνο τις κορυφαίες 1000 λέξεις

# Εφαρμογή του TfidfVectorizer στα tokens_list
tfidf_matrix = tfidf_vectorizer.fit_transform([" ".join(filtered_tokens) for filtered_tokens in tokens_list])

# Εκτύπωση του σχήματος του tfidf_matrix
print("Shape of TF-IDF matrix:", tfidf_matrix.shape)


# β ερωτημα 

# Κανονικοποίηση του TF-IDF matrix
scaler = MaxAbsScaler()
tfidf_matrix_normalized = scaler.fit_transform(tfidf_matrix)

# Κανονικοποίηση των χαρακτηριστικών date_min και date_max
data[['date_min', 'date_max']] = scaler.fit_transform(data[['date_min', 'date_max']])


# Εκτύπωση του σχήματος του κανονικοποιημένου tfidf_matrix
print("Shape of normalized TF-IDF matrix:", tfidf_matrix_normalized.shape)




#γ ερωτημα 



# Διαχωρισμός των δεδομένων σε σύνολα εκπαίδευσης και ελέγχου με χρήση K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

 # Αποθηκεύστε τα RMSE για κάθε fold
rmse_scores = []

for kf, (train_index, test_index) in enumerate(kf.split(tfidf_matrix_normalized)):
    X_train, X_test = tfidf_matrix_normalized[train_index], tfidf_matrix_normalized[test_index]
    y_train, y_test = data[['date_min', 'date_max']].iloc[train_index], data[['date_min', 'date_max']].iloc[test_index]

#A2 


   # Εκπαίδευση του ΤΝΔ
    model = MLPRegressor(hidden_layer_sizes=(10), max_iter=1000, activation='relu', learning_rate_init=0.001, random_state=42)
    model.fit(X_train, y_train)


  # ΕΡΩΤΗΜΑ Α2
    
    # Πρόβλεψη
    y_pred = model.predict(X_test)
    
    # Υπολογισμός RMSE
    rmse = np.sqrt(np.mean(np.square(custom_error(y_test, y_pred))))

  # Αποθηκεύστε το RMSE για το τρέχον fold
    rmse_scores.append(rmse)


    model.out_activation_ = 'identity'
    
     # Παρουσίαση γραφικών παραστάσεων για τη σύγκλιση
    train_errors = np.zeros(model.max_iter)
    for i in range(model.max_iter):
            model.partial_fit(X_train, y_train)
            train_errors[i] = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
    plt.plot(range(1, len(train_errors) + 1), train_errors, label=f'{kf}')


    print(f"Fold {kf+1}: RMSE = {rmse}")

# Υπολογίστε τον μέσο όρο των RMSE για όλα τα folds
mean_rmse = np.mean(rmse_scores)
print(f"Overall RMSE with (10) hidden neurons = {mean_rmse}")

# Προσθήκη ετικετών και τίτλου στο γράφημα
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.title('Convergence Rate (Mean RMSE per Epoch)')
plt.legend()
plt.show()
