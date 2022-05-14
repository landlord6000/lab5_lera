import numpy as np
import matplotlib.pyplot as plt

a = 0.0134
b = 1
c = 4.35 * (10**(-4))
q = 1.5*1000
alpha0 = 1.94 * 10**(-2)
gamma = 0.20 * 10**(-2)
R = 0.5
T0k = 300
betta = 300
Fc = -10

def f0(h, T0, T1):
    df0_dx0 = a * (b + 2*c*T0)
    df0_dx1 = -a * (b + 2*c*T1)
    
    return df0_dx0, df0_dx1

def fn(h, Tnm, Tn):
    dfn_dxnm = a * (b + c*Tn)
    dfn_dxn = -((a*q**4*(b + 2*c*Tn - c*Tnm) + h*(q - Tn)**3*alpha0*(q - 5*Tn + 4*betta)+h*q**4*gamma)/q**4)
    
    return dfn_dxnm, dfn_dxn
    
def f(h, Tnm, Tn, Tnp):
    df_dxnm = (a*(b + c*Tnm))/h
    df_dxn = -((2*(a*q**4*R*(b + c*Tn) + h**2*((q + 4*T0k - 5*Tn)*(q - Tn)**3*alpha0 + q**4*gamma)))/(h*q**4*R))
    df_dxnp = (a*(b + c*Tnp))/h
    
    return df_dxnm, df_dxn, df_dxnp
    
def find_acc(x0, x1):
    maxx = 0
    for i in range(len(x0)):
        if abs(x0[i] - x1[i]) > maxx:
            maxx = abs(x0[i] - x1[i])
    return maxx

def create_matr_F(n, x0):
    matr = []
    h = 10 / n
    
    l = []
    df0_dx0, df0_dx1 = f0(h, x0[0], x0[1])
    l.append(df0_dx0); l.append(df0_dx1)
    for i in range(n-1):
        l.append(0)
    matr.append(l)
    
    for i in range(1, n):
        l = [0]*(i-1)
        df_dxnm, df_dxn, df_dxnp = f(h, x0[i-1], x0[i], x0[i+1])
        l.append(df_dxnm); l.append(df_dxn); l.append(df_dxnp);
        for j in range(i+2, n+1):
            l.append(0)

        matr.append(l)

    l = [0]*(n-1)
    dfn_dxnm, dfn_dxn = fn(h, x0[n-1], x0[n])
    l.append(dfn_dxnm); l.append(dfn_dxn)
    matr.append(l)

    return matr

def f_x0(n, x0):
    h = 10 / n
    T0, T1 = x0[0], x0[1]
    Tnm1, Tn1 = x0[n-1], x0[n]

    f_ = []
    f_.append((a*b + a*c*T0)*T0 - (a*b + a*c*T1)*T1 - h * Fc)

    for i in range(1,n):
        Tnm, Tn, Tnp = x0[i-1], x0[i], x0[i+1]
        fn = (2*a*b + a*c*Tnm + a*c*Tn )/(2*h)* Tnm - ((2*a*b + a*c*Tnm + a*c*Tn)/(2*h) + \
        (2*a*b + a*c*Tnp + a*c*Tn)/(2*h) + 2/R*(alpha0*(Tn/q - 1)**4 + gamma) * h) * Tn + \
        (2*a*b + a*c*Tnp + a*c*Tn)/(2*h)* Tnp + 2*T0k/R*(alpha0*(Tn/q - 1)**4 + gamma)*h
        f_.append(fn)

    f_.append((a*b + a*c*Tn)*Tnm1 - ((a*b + a*c*Tn1) + (alpha0*(Tn1/q - 1)**4 + gamma)*h)* \
             Tn1 + (alpha0*(Tn1/q - 1)**4 + gamma)*h*betta)

    return f_



n = 50
x0 = [0]*(n+1)
x1 = [300]*(n+1)

while (find_acc(x0, x1) >= 1e-6):
    x0 = x1
    matr = create_matr_F(n, x0)
    matr = np.array(create_matr_F(n, x0))
    f_ = f_x0(n, x0)
    f_ = np.array(f_x0(n, x0))
    x0 = np.array(x0)

    x1 = (x0 - np.dot(np.linalg.inv(matr),f_)).tolist()
    x0 = x0.tolist()
    #print(find_acc(x0, x1))


x1 = [x_ for x_ in x1]
#print(x1)

plt.title('Зависимость T от x')
plt.plot([i for i in range(n+1)], x1,'g-')
plt.grid(True)
plt.show()


