# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 14:57:22 2020

@author: aditya
"""
import math
import matplotlib.pyplot as plt
from scipy import optimize,interpolate
import numpy as np

def price(ytm,ttm,freq,cpn_rate,fv=100. ):
    """
    Compute the bond price
    
    Parameters
    ----------
    cpn_rate : float
        Annualized coupon rate of a bond as percentage of face value
    ttm : int
        Time to Maturity in years.
    ytm : float
        annulaized yield to maturity.
    fv : float
        Face value of bond.
    freq : int
        frequency of payment.

    Returns
    -------
    price: float
         Bond Price

    """
    if freq == 0: # zc bond
        return fv/((1+ytm)**ttm)
    
    cpn = cpn_rate/freq*fv
    periods = freq*ttm
    
    dfs = [ (1+(ytm/freq))**(n*-1) for n in range(1,periods+1) ] # discount factors based on ytm
    annuity = sum([cpn*df for df in dfs])
    fv_price = fv*dfs[-1]
    return annuity+fv_price

def yield_delta(ytm,*args, **kwrgs):
    """
    yield duration = ( dP/dY)
        where P = Bond Price
              Y = Yield to Maturity

    Parameters
    ----------
    args : tuple
        Same parameter as bond_price : ytm,ttm,freq,cpn_rate,fv=100. 

    Returns
    -------
    bond duration
    """
    del_y = 0.0001
    p_up = price(ytm+del_y, *args, **kwrgs)
    p_dn = price(ytm-del_y, *args, **kwrgs)
    delta = (p_up-p_dn)/(2*del_y)
    return delta

def mod_yield_duration(ytm,*args, **kwrgs):
    """
    Modified yield duration = -1/P * ( dP/dY)
        where P = Bond Price
              Y = Yield to Maturity

    Parameters
    ----------
    args : tuple
        Same parameter as bond_price : ytm,ttm,freq,cpn_rate,fv=100. 

    Returns
    -------
    bond duration

    """
    p = price(ytm, *args, **kwrgs)
    duration = -(1/p)*yield_delta(ytm, *args, **kwrgs)
    return duration
    

def convexity(ytm,*args, **kwrgs):
    """
    convexity = ( dD/dY)
        where D = Delta
              Y = Yield to Maturity

    Parameters
    ----------
    args : tuple
        Same parameter as bond_price : ytm,ttm,freq,cpn_rate,fv=100. 

    Returns
    -------
    bond convexity

    """
    del_y = 0.0001
    scaling_factor = del_y
    del_up = yield_delta(ytm+del_y, *args, **kwrgs)
    del_dn = yield_delta(ytm-del_y, *args, **kwrgs)
    convexity = (del_up-del_dn)/(2*del_y)
    return convexity*scaling_factor


def theta(ytm,ttm,freq,cpn_rate,fv=100. ):
    p = price(ytm,ttm,freq,cpn_rate,fv)
    dt = -1/252
    p_new = price(ytm,ttm+dt,freq,cpn_rate,fv)
    return p_new-p
    
def calc_ytm(price,ttm,freq,cpn, fv=100.):
    """
    Yield to maturity is calculated based on below formula
        P = (c/y) *[1 - 1/(1+y)**n] + 100/(1+y)**n

    Parameters
    ----------
    price : price of bond
    ttm = time to maturity in years
    freq = freq of payment
    cpn : coupon per period
    fv : future value or the final price

    Returns
    -------
    yield to maturity

    """
    price = float(price)
    ttm = float(ttm)
    cpn = float(cpn)
    fv = float(fv)
    
    if freq == 0: #zero cpn bond
        return (fv/price)**(1/ttm) -1
    
    cpn = cpn/freq # coupon per period
    n = ttm*freq # no. of periods
    
    def func(y):
        return price - (cpn/y)*(1 - (1+y)**-n) - fv*(1+y)**-n
    
    ytm = optimize.newton(func, 0.01, tol=.0001, maxiter=100)
    return ytm*freq
    

def interpolate_ytm(ttm_list, ytm_list, max_tnr=49):
    interp_ytm = list()
    interp = interpolate.interp1d(ttm_list, ytm_list, bounds_error=False, fill_value=np.nan)

    for tnr in range(1,max_tnr):
        value = float(interp(tnr))
        if not np.isnan(value):
            interp_ytm.append(value)
    return interp_ytm

def plot(**kwargs):
    plt.figure(figsize=(12,8))
    series = kwargs['series']
    xlabel = kwargs['xlabel']
    ylabel = kwargs['ylabel']
    for x,data,label in series:
        plt.plot(x,data,lw=1.5,label=label)
    plt.grid(True)
    plt.legend(loc=0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

if __name__ == '__main__':
    print(price(0.01,1,2,0.06))
    assert math.isclose(price(0.06,5,2,0.06), 100. )
    assert price(0.03,5,2,0.06) > 100.
    assert price(0.06,5,2,0.03) < 100.
    print(mod_yield_duration(0.08,20,1,.12))
                 
    
    