import numbers
import numpy as np
import pandas as pd     
import matplotlib.pyplot as plt
#import skfuzzy as fuzz

from itertools import product
from itertools import combinations

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import KBinsDiscretizer
from itertools import product
from sklearn.metrics import make_scorer

#test
class DMC(BaseEstimator, ClassifierMixin):
    _parameter_constraints: dict = {
        "N": [Interval(numbers.Integral, 1, None, closed="left")],
        "T": [Interval(numbers.Integral, 2, None, closed="left"), StrOptions({"auto"})],
        "m": [Interval(numbers.Real,1,None, closed="neither")],
        "discretization": [StrOptions({"kmeans", "DT", "KBins", "cmeans"})],
        "L": ["array-like", None],
        "box": ["array-like", None],
        "random_state": ["random_state"],
        }
    def __init__(
            self,
            T="auto",
            m=1.5,
            N=10000,
            discretization='kmeans', 
            L=None,
            box=None,
            random_state=None,
            predict_with_probabilites=False,
            discretization_model=None,
    ):
        """
        Initialize the DMC model.

        Parameters:
        N : int, default=1000
            Maximum number of iterations for the algorithm
        T : int or str, default='auto'
            Number of  discrete profiles. Must be an integer greater than or equal to 2. or 'auto' for automatic determination.
        discretization : str, default='kmeans'
            Method of discretization to use. Must be 'kmeans' or 'DT'(decision tree).
        L : array-like or None, default=None
            Loss function, default is zero-one loss.
        box : array-like or None, default=None
            Box constraints for the piStar.
        random_state : int, RandomState instance or None, default=None
            Seed for random number generator for reproducibility.
        """
        self.piStar = None
        self.piTrain = None
        self.pHat = None 
        self.discretization_model=discretization_model
        self.T = T
        self.N = N
        self.L = L
        self.m = m
        self.discretization = discretization
        self.box = box
        self.label_encoder = LabelEncoder()

        self.random_state = random_state
        self.predict_with_probabilites=predict_with_probabilites
        self._validate_params()

    def fit(self, X, y, **paramT):
        self.random_state = check_random_state(self.random_state)
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.DataFrame): # Convert y to a numpy array if is a Dataframe
            y = y.to_numpy().ravel()  # Use ravel() to make sure that y is one-dimensional

        y_encoded = self.label_encoder.fit_transform(y)
        K = len(np.unique(y_encoded))
        if self.L is None:
            self.L = np.ones((K, K)) - np.eye(K)

        if self.discretization == 'kmeans':
            if self.discretization_model==None:
                if self.T == 'auto':
                    self.T = self.get_T_optimal(X, y_encoded, **paramT)['T']
                self.discretization_model = KMeans(n_clusters=self.T,random_state=self.random_state,
                                               init="random")
                self.discretization_model.fit(X)
            self.discrete_profiles = self.discretization_model.labels_
            self.pHat = compute_pHat(self.discrete_profiles, y_encoded, K, self.T)
        if self.discretization == "DT":
            self.discretization_model = DecisionTreeClassifier(ccp_alpha=0, 
                                           class_weight="balanced", 
                                           criterion='entropy', 
                                           min_samples_leaf=15,
                                        max_features= 'sqrt',
                                           splitter='best',
                                           random_state=self.random_state).fit(X, y_encoded)
            
            self.discrete_profiles=self.discretisation_DT(X, self.discretization_model)
            self.T=self.discretization_model.get_n_leaves()
            self.pHat = compute_pHat(self.discrete_profiles, y_encoded, K, self.T)
            
        self.piTrain = compute_pi(y_encoded, K)
        self.piStar, rStar, self.RStar, V_iter, stockpi = compute_piStar(self.pHat, y_encoded, K, self.L, self.T, self.N, 0, self.box)
        self._is_fitted = True
        self.classes_ = np.unique(y_encoded)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X, pi=None):
        check_is_fitted(self, ['X_', 'y_', 'classes_'])
        if pi is None:
            pi = self.piStar
        #print(predict_profile_label(pi, self.pHat, self.L))
        if self.discretization=="kmeans":
            discrete_profiles=self.discretization_model.predict(X)

        elif self.discretization=="kmeans" and self.predict_with_probabilites==True:
                num_instances = discrete_profiles.shape[0]
                Yhat = np.zeros((num_instances, 1), dtype=int)
                OutputProba=self.predict_prob(X)
                for i in range(num_instances):
                    Yhat[i, 0] = np.argmax(np.random.multinomial(1, OutputProba[i, :]))


        elif self.discretization=="DT":
            discrete_profiles=self.discretisation_DT(X, self.discretization_model)


        return self.label_encoder.inverse_transform(
            predict_profile_label(pi, self.pHat, self.L)[discrete_profiles]
        )

    def predict_prob(self, X, pi=None):
        #I think we have to change this
        check_is_fitted(self)
        if pi is None:
            pi = self.piStar
        if self.discretization == 'kmeans':
            lambd = (pi.reshape(-1, 1) * self.L).T @ self.pHat
            prob = lambd / np.sum(lambd, axis=0)
            return prob[:, self.discretization_model.predict(X)].T

    def get_T_optimal(self, X, y, T_start=5, T_end=100, Num_t_Values=50):
        param_grid = {
            'T': np.linspace(T_start, T_end, Num_t_Values, dtype=int)
        }
        grid_search = GridSearchCV(estimator=self, param_grid=param_grid, cv=2,scoring=gloabl_risk, error_score='raise')
        grid_search.fit(X, y)
        return grid_search.best_params_
        
    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        return {"T": self.T, "N": self.N,"discretization":self.discretization,"L":self.L,
                "random_state":self.random_state,"box":self.box}
       
    def discretisation_DT(self,X, modele) :
        '''
        Parameters
        ----------
        X : DataFrame
        Features.
        modele : Decision Tree Classifier Model
        Decidion Tree model.

        Returns
        -------
        Xdiscr : Vector
            Discretised features.

        '''
        Xdiscr = DecisionTreeClassifier.apply(modele, X, check_input=True)
         # Obtener los índices únicos y su inversa
        valores_unicos, inversa = np.unique(Xdiscr, return_inverse=True)
    
        # Crear un mapeo de índices únicos a valores enteros consecutivos
        mapeo = {valor: indice  for indice, valor in enumerate(valores_unicos)}
    
        # Mapear los valores originales de Xdiscr a sus equivalentes enteros consecutivos
        Xdiscr_enteros = np.array([mapeo[valor] for valor in Xdiscr])
        return Xdiscr_enteros



def compute_pi(y: np.ndarray, K: int):
    """
    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Labels

    K : int
        Number of classes

    Returns
    -------
    pi : ndarray of shape (K,)
        Proportion of classes
    """
    pi = np.zeros(K)
    total_count = len(y)

    for k in range(K):
        pi[k] = np.sum(y == k) / total_count
    return pi


def compute_pHat(XD: np.ndarray, y: np.ndarray, K: int, T: int):
    """
    Parameters
    ----------
    XD : ndarray of shape (n_samples,)
        Labels of profiles for each data point

    y : ndarray of shape (n_samples,)
        Labels

    K : int
        Number of classes

    T : int
        Number of profiles

    Returns
    -------
    pHat : ndarray of shape(K, n_profiles)
    """
    pHat = np.zeros((K, T))

    for k in range(K):
        Ik = np.where(y == k)[0]
        mk = len(Ik)
        pHat[k] = np.bincount(XD[Ik], minlength=T)/mk
        #Count number of occurrences of each value in array of non-negative ints.
    return pHat

def delta_proba_U(U, pHat, pi, L, methode='before', temperature=0):
    '''
    Parameters
    ----------
    U : Array

    pHat : Array of floats
        Probability estimate of observing the features profile.
    pi : Array of floats
        Real class proportions.
    L : Array
        Loss function.

    Returns
    -------
    Yhat : Vector
        Predicted labels.
    '''

    def softmin_with_temperature(X, temperature=1.0, axis=1):
        X = -X
        X_max = np.max(X, axis=axis, keepdims=True)
        X_adj = X - X_max

        exp_X_adj = np.exp(X_adj / temperature)
        softmax_output = exp_X_adj / np.sum(exp_X_adj, axis=axis, keepdims=True)

        return softmax_output

    lambd = U.T @ ((pi.T * L).T @ pHat).T

    if methode == 'softmin':
        prob = softmin_with_temperature(lambd, temperature)

    elif methode == 'argmin':
        prob = np.zeros_like(lambd)
        rows = np.arange(lambd.shape[0])
        cols = np.argmin(lambd, axis=1)
        prob[rows, cols] = 1

    elif methode == 'proportion':
        prob = 1 - np.divide(lambd, np.sum(lambd, axis=1)[:, np.newaxis])

    elif methode == 'before':
        prob = 1 - np.divide(lambd, np.sum(lambd, axis=1)[:, np.newaxis])

    elif methode == 'after':
        prob_init = 1 - np.divide(lambd, np.sum(lambd, axis=1)[:, np.newaxis])
        index = np.argmax(prob_init, axis=1)
        prob = np.zeros_like(prob_init)
        prob[np.arange(index.shape[0]), index] = 1
    return prob


def compute_conditional_risk(y_true: np.ndarray, y_pred: np.ndarray, K: int, L: np.ndarray):
    '''
    Function to compute the class-conditional risks.
    Parameters
    ----------
    YR : DataFrame
        Real labels.
    Yhat : Array
        Predicted labels.
    K : int
        Number of classes.
    L : Array
        Loss Function.

    Returns
    -------
    R : Array of floats
        Conditional risks.
    confmat : Matrix
        Confusion matrix.
    '''
    Labels=[i for i in range(K)]
    confmat=confusion_matrix(np.array(y_true),np.array(y_pred),normalize='true',labels=Labels)
    R=np.sum(np.multiply(L, confmat),axis=1)
    
    return R, confmat

def score_global_risk(y_true,y_pred):
    k=len(np.unique(y_true))
    L = np.ones((k, k)) - np.eye(k)
    pi=compute_pi(y_true, k)
    R,M=compute_conditional_risk(y_true, y_pred, k, L)
    score=np.sum(R * pi)
    return score

gloabl_risk=make_scorer(score_global_risk,greater_is_better=False)

def max_risk(y_true,y_pred):
    k=len(np.unique(y_true))
    L = np.ones((k, k)) - np.eye(k)
    pi=compute_pi(y_true, k)
    R,M=compute_conditional_risk(y_true, y_pred, k, L)


def compute_global_risk(R, pi):
    """
    Parameters
    ----------
    R : ndarray of shape (K,)
        Conditional risk
    pi : ndarray of shape (K,)
        Proportion of classes

    Returns
    -------
    r : float
        Global risk.
    """

    r = np.sum(R * pi)

    return r


def predict_profile_label(pi, pHat, L):
    lambd = (pi.reshape(-1, 1) * L).T @ pHat
    lbar = np.argmin(lambd, axis=0)
    return lbar

def proj_simplex_Condat(K, pi):
    """
    This function is inspired from the article: L.Condat, "Fast projection onto the simplex and the 
    ball", Mathematical Programming, vol.158, no.1, pp. 575-585, 2016.
    Parameters
    ----------
    K : int
        Number of classes.
    pi : Array of floats
        Vector to project onto the simplex.

    Returns
    -------
    piProj : List of floats
        Priors projected onto the simplex.

    """

    linK = np.linspace(1, K, K)
    piProj = np.maximum(pi - np.max(((np.cumsum(np.sort(pi)[::-1]) - 1) / (linK[:]))), 0)
    piProj = piProj / np.sum(piProj)
    return piProj

def graph_convergence(V_iter):
    '''
    Parameters
    ----------
    V_iter : List
        List of value of V at each iteration n.

    Returns
    -------
    Plot
        Plot of V_pibar.

    '''

    figConv = plt.figure(figsize=(8, 4))
    plt_conv = figConv.add_subplot(1, 1, 1)
    V = V_iter.copy()
    V.insert(0, np.min(V))
    font = {'weight': 'normal', 'size': 16}
    plt_conv.plot(V, label='V(pi(n))')
    plt_conv.set_xscale('log')
    plt_conv.set_ylim(np.min(V), np.max(V) + 0.01)
    plt_conv.set_xlim(10 ** 0)
    plt_conv.set_xlabel('Interation n', fontdict=font)
    plt_conv.set_title('Maximization of V over U', fontdict=font)
    plt_conv.grid(True)
    plt_conv.grid(which='minor', axis='x', ls='-.')
    plt_conv.legend(loc=2, shadow=True)

def num2cell(a):
    if type(a) is np.ndarray:
        return [num2cell(x) for x in a]
    else:
        return a

def proj_onto_polyhedral_set(pi, Box, K) :
    '''
    Parameters
    ----------
    pi : Array of floats
        Vector to project onto the box-constrained simplex.
    Box : Array
        {'none', matrix} : Box-constraint on the priors.
    K : int
        Number of classes.

    Returns
    -------
    piStar : Array of floats
            Priors projected onto the box-constrained simplex.

    '''
    
    # Verification of constraints
    for i in range(K) :
        for j in range(2) :
            if Box[i,j] < 0 :
                Box[i,j] = 0
            if Box[i,j] > 1 :
                Box[i,j] = 1

    # Generate matrix G:
    U = np.concatenate((np.eye(K), -np.eye(K), np.ones((1,K)), -np.ones((1,K))))            
    eta = Box[:,1].tolist() + (-Box[:,0]).tolist() + [1] + [-1]

    n = U.shape[0]
    
    G = np.zeros((n,n))
    for i in range(n) :
        for j in range(n) :
            G[i,j] = np.vdot(U[i,:],U[j,:])
    
    
    # Generate subsets of {1,...,n}:
    M = (2**n)-1
    I = num2cell(np.zeros((1,M)))
    
    i = 0
    for l in range(n) :
        T = list(combinations(list(range(n)), l+1))
        for p in range(i,i+len(T)) :
            I[0][p] = T[p-i]
        i = i+len(T)
            
        
    # Algorithm    
        
    for m in range(M) :
        Im = I[0][m]
 
        Gmm = np.zeros((len(Im), len(Im)))
        ligne = 0
        for i in Im :
            colonne = 0
            for j in Im :
                Gmm[ligne,colonne] = G[i,j]
                colonne += 1
            ligne +=1
        

        if np.linalg.det(Gmm)!=0 :
            
            nu = np.zeros((2*K+2,1))
            w = np.zeros((len(Im),1))
            for i in range(len(Im)) :
                w[i] = np.vdot(pi,U[Im[i],:]) - eta[Im[i]]
            
            S = np.linalg.solve(Gmm,w) 
            
            for e in range(len(S)) :
                nu[Im[e]] = S[e]
            
            
            if np.any(nu<-10**(-10)) == False  :
                A = G.dot(nu)
                z = np.zeros((1,2*K+2))
                for j in range(2*K+2) :
                    z[0][j] = np.vdot(pi,U[j,:]) - eta[j] - A[j]
                    
                    
                if np.all(z<=10**(-10)) == True :
                    pi_new = pi
                    for i in range(2*K+2) :
                        pi_new = pi_new - nu[i]*U[i,:]

    piStar = pi_new

    # Remove noisy small calculus errors:
    piStar = piStar/piStar.sum()
    
    return piStar

def proj_onto_U(pi, Box, K) :
    '''
    Parameters
    ----------
    pi : Array of floats
        Vector to project onto the box-constrained simplex..
    Box : Matrix
        {'none', matrix} : Box-constraint on the priors.
    K : int
        Number of classes.

    Returns
    -------
    pi_new : Array of floats
            Priors projected onto the box-constrained simplex.

    '''
    
    check_U = 0
    if pi.sum() ==1 :
        for k in range(K) :
            if (pi[0][k] >= Box[k,0]) & (pi[0][k] <= Box[k,1]) :
                check_U = check_U + 1
    
    if check_U == K :
        pi_new = pi

      
    if check_U < K :
        pi_new = proj_onto_polyhedral_set(pi, Box, K)
    
    return pi_new



def compute_piStar(pHat, y_train, K, L, T, N, optionPlot, Box):
    """
    Parameters
    ----------
    pHat : Array of floats
        Probability estimate of observing the features profile in each class.
    y_train : Dataframe
        Real labels of the training set.
    K : int
        Number of classes.
    L : Array
        Loss Function.
    T : int
        Number of discrete profiles.
    N : int
        Number of iterations in the projected subgradient algorithm.
    optionPlot : int {0,1}
        1 plots figure,   0: does not plot figure.
    Box : Array
        {'none', matrix} : Box-constraints on the priors.

    Returns
    -------
    piStar : Array of floats
        Least favorable priors.
    rStar : float
        Global risks.
    RStar : Array of float
        Conditional risks.
    V_iter : Array
        Values of the V function at each iteration.
    stockpi : Array
        Values of pi at each iteration.

    """
    # IF BOX-CONSTRAINT == NONE (PROJECTION ONTO THE SIMPLEX)
    if Box is None:
        pi = compute_pi(y_train, K).reshape(1, -1)
        rStar = 0
        piStar = pi
        RStar = 0

        V_iter = []
        stockpi = np.zeros((K, N))

        for n in range(1, N + 1):
            # Compute subgradient R at point pi (see equation (21) in the paper)
            lambd = np.dot(L, pi.T * pHat)
            R = np.zeros((1, K))

            mu_k = np.sum(L[:, np.argmin(lambd, axis=0)] * pHat, axis=1)
            R[0,:] = mu_k
            stockpi[:,n-1] = pi[0,:]

            r = compute_global_risk(R, pi)
            V_iter.append(r)
            if r > rStar:
                rStar = r
                piStar = pi
                RStar = R
                # Update pi for iteration n+1
            gamma = 1 / n
            eta = np.maximum(float(1), np.linalg.norm(R))
            w = pi + (gamma / eta) * R
            pi = proj_simplex_Condat(K, w)

        # Check if pi_N == piStar
        lambd = np.dot(L, pi.T * pHat)
        R = np.zeros((1, K))

        mu_k = np.sum(L[:, np.argmin(lambd, axis=0)] * pHat, axis=1)
        R[0,:] = mu_k
        stockpi[:,n-1] = pi[0,:]

        r = compute_global_risk(R, pi)
        if r > rStar:
            rStar = r
            piStar = pi
            RStar = R

        if optionPlot == 1:
            print("si")
            graph_convergence(V_iter)

    # IF BOX-CONSTRAINT
    if Box is not None:
        pi = compute_pi(y_train, K).reshape(1, -1)
        rStar = 0
        piStar = pi
        RStar = 0

        V_iter = []
        stockpi = np.zeros((K, N))

        for n in range(1, N + 1):
            # Compute subgradient R at point pi (see equation (21) in the paper)
            lambd = np.dot(L, pi.T * pHat)
            R = np.zeros((1, K))

            mu_k = np.sum(L[:, np.argmin(lambd, axis=0)] * pHat, axis=1)
            R[0,:] = mu_k
            stockpi[:,n-1] = pi[0,:]

            r = compute_global_risk(R, pi)
            V_iter.append(r)
            if r > rStar:
                rStar = r
                piStar = pi
                RStar = R
                # Update pi for iteration n+1
            gamma = 1 / n
            eta = np.maximum(float(1), np.linalg.norm(R))
            w = pi + (gamma / eta) * R
            pi = proj_onto_U(w, Box, K)

        # Check if pi_N == piStar
        lambd = np.dot(L, pi.T * pHat)
        R = np.zeros((1, K))

        mu_k = np.sum(L[:, np.argmin(lambd, axis=0)] * pHat, axis=1)
        R[0,:] = mu_k
        stockpi[:,n-1] = pi[0,:]
            
        r = compute_global_risk(R, pi)
        if r > rStar:
            rStar = r
            piStar = pi
            RStar = R

        if optionPlot == 1:
            print("AQU")
            graph_convergence(V_iter)

    return piStar, rStar, RStar, V_iter, stockpi


import copy
class BinaryRelevanceDMC():
    def __init__(self,T=50,random_state=None):
        self.T=T
        self.random_state = random_state

    def inicializar(self, X,Y):
        discretization_model = KMeans(n_clusters=self.T,random_state=self.random_state,
                                               init="random")
        
        discretization_model.fit(X)
        self.classifiers = [copy.deepcopy(DMC(discretization_model=discretization_model,T=self.T))
                             for _ in range(Y.shape[1])]

    def fit(self, X, Y):
        self.inicializar(X,Y)
        partition = list(range(Y.shape[1]))
        for label in partition:
            Y_subset = np.copy(Y[:, label])
            self.classifiers[label].fit(X, Y_subset.ravel())

    def predict(self, X_test):
        predictions = []
        for model in self.classifiers:
            predictions.append(model.predict(X_test))
        resultado = np.array(predictions).T
        return resultado