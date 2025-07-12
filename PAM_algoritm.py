def cluster_PAM(X, k):
    import pandas as pd
    import numpy as np
    # Funkcja służąca do przeprowadzania klasteryzacji metodą k-medoidów - algorytm PAM
    # Przyjmuje dwa argumenty:
    # X - Ramka danych, gdzie każdy wiersz to jedna obserwacja, a kolejne kolumny to współrzędne
    # k - Liczba klastrów, na które chcemy podzielić zbiór (liczba całkowita, większa od 1, mniejsza od liczby wierszy X)

    # Sprawdzenie poprawności argumentów
    if type(X) is not pd.DataFrame:
        raise TypeError("X musi być ramką danych!")
    if type(k) is not int:
        raise TypeError("k musi być liczbą całkowitą!")
    if k < 1 or k > len(X):
        raise ValueError("k poza odpowiednim zakresem!")
    
    # Funkcja do obliczania macierzy odległości pomiędzy punktami i przypisania ich do odpowiednich klastrów
    def oblicz_macierz(X, medoids):
        distances = np.linalg.norm(X.values[:, np.newaxis] - medoids.values, axis=2) # Obliczamy odległości (euklidesowe) pomiędzy punktami a medoidami
        cluster_labels = np.argmin(distances, axis=1) # Przypisania klastra do każdego punktu
        total_distance = np.sum(np.min(distances, axis=1)) #Oblicza sumę odległośći punktów od ich medoidów
        return cluster_labels, total_distance
    
    # FAZA BUDOWY

     # Wybieramy losowo k obserwacji jako nasze początkowe medoidy
    medoids = X.sample(k)

    # Obliczamy początkową macierz odległości pomiędzy punktami i przypisujemy je do odpowiednich klastrów
    cluster_labels, cur_sum = oblicz_macierz(X, medoids)

    # FAZA ZMIANY

    while True:
        iter = 0 # Zabezpieczenie przed nieskończoną pętlą
        zmiana = False # Zmienna, która będzie informować o tym czy nastąpiła zmiana medoidów
        for i in range(k):
            cluster_points = X[cluster_labels == i].copy() # Wybieramy punkty z danego klastra

            # Testowanie wszystkich punktów w klastrze jako potencjalnych nowych medoidów
            for j in cluster_points.index:
                pot_medoids = medoids.copy()
                pot_medoids.iloc[i] = X.loc[j]
                
                # Zaktualizowanie odległości dla wszystkich punktów 
                pot_labels, pot_sum = oblicz_macierz(X, pot_medoids)
                
                # Jeśli nowa konfiguracja daje mniejszą sumę odległości, akceptujemy ją za aktualną
                if pot_sum < cur_sum:
                    cur_sum = pot_sum
                    cluster_labels = pot_labels.copy()
                    medoids = pot_medoids
                    zmiana = True
                    iter += 1
        if zmiana == False or iter > 10000: # Przerywamy działania funkcji, gdy nie nastąpiła zmiana medoidów lub przekroczono limit iteracji
            break
    
    return medoids, cluster_labels