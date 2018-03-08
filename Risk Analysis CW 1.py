import numpy as np
import pandas as pd
import time
import random
import stat
from scipy.stats import norm
import scipy.stats as stats
from scipy.interpolate import interp1d
import pylab
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.utils import resample
from statsmodels.graphics.tsaplots import plot_acf
from datetime import date

np.set_printoptions(threshold=np.inf)


VZ = pd.read_csv('/Users/kingf.wong/Downloads/Risk_Analysis_CW/VZ.csv',header=0)
INTC = pd.read_csv('/Users/kingf.wong/Downloads/Risk_Analysis_CW/INTC.csv',header=0)
JPM = pd.read_csv('/Users/kingf.wong/Downloads/Risk_Analysis_CW/JPM.csv',header=0)
AAPL = pd.read_csv('/Users/kingf.wong/Downloads/Risk_Analysis_CW/AAPL.csv',header=0)
MSFT = pd.read_csv('/Users/kingf.wong/Downloads/Risk_Analysis_CW/MSFT.csv',header=0)
PG = pd.read_csv('/Users/kingf.wong/Downloads/Risk_Analysis_CW/PG.csv',header=0)

#   Calculate portfolio value throughout time

current_port_V = np.array(VZ['Adj Close'])*20 + np.array(INTC['Adj Close'])*10 + np.array(JPM['Adj Close'])*10 \
                 + np.array(AAPL['Adj Close'])*10 + np.array(MSFT['Adj Close'])*4 + np.array(PG['Adj Close'])*-5

d = {'VZ':np.array(VZ['Adj Close'])*20, 'INTC':np.array(INTC['Adj Close'])*10, 'JPM':np.array(JPM['Adj Close'])*10
    ,'AAPL':np.array(AAPL['Adj Close'])*10, 'MSFT':np.array(MSFT['Adj Close'])*4, 'PG':np.array(PG['Adj Close'])*-5}
    #,'Current Port Value': current_port_V}
ds = {'1 VZ':np.array(VZ['Adj Close']), '2 INTC':np.array(INTC['Adj Close']), '3 JPM':np.array(JPM['Adj Close'])
    ,'4 AAPL':np.array(AAPL['Adj Close']), '5 MSFT':np.array(MSFT['Adj Close']), '6 PG':np.array(PG['Adj Close'])}



#Option = pd.read_csv('/Users/kingf.wong/Downloads/Risk_Analysis_CW/CwkData201728.xlsx',header=0)

dates = np.array(VZ['Date'])        # takes dates reference  from VZ.csv

# d = {'VZ':np.array(VZ['Adj Close']), 'INTC': np.array(INTC['Adj Close']), 'JPM':np.array(JPM['Adj Close']),
#      'AAPL': np.array(AAPL['Adj Close']), 'MSFT': np.array(MSFT['Adj Close']), 'PG': np.array(PG['Adj Close']),
#      'Port_Value': current_port_V}
#
# df = pd.DataFrame(data=d, index=dates)

#build dataframe of stock closing price, index with trading dates, stock columns now arranged alphabetically in dg
# dg dataframe contains portfolio stocks with weights
# dspd dataframe contains price information of portfolio stocks

dg = pd.DataFrame(data=d, index=dates)
dspd = pd.DataFrame(data=ds, index=dates)
# print(dspd)

# last day 6 stocks value
last_day_port_v = np.array(dspd[dspd.index.get_loc('2018-02-09'):])

# print(last_day_port_v)

dg.to_csv('price.csv', encoding='utf-8', index=True)

########    calculate daily log return from Adj Close  ###########

VZ_log_ret_close = np.diff(np.log(np.array(VZ['Adj Close'])))
INTC_log_ret_close = np.diff(np.log(np.array(INTC['Adj Close'])))
JPM_log_ret_close = np.diff(np.log(np.array(JPM['Adj Close'])))
AAPL_log_ret_close = np.diff(np.log(np.array(AAPL['Adj Close'])))
MSFT_log_ret_close = np.diff(np.log(np.array(MSFT['Adj Close'])))
PG_log_ret_close = np.diff(np.log(np.array(PG['Adj Close'])))


###### return dates start on 0
ret_dates = dates[1:len(dates)]

d_ret = {'VZ':VZ_log_ret_close, 'INTC': INTC_log_ret_close, 'JPM':JPM_log_ret_close , 'AAPL': AAPL_log_ret_close, 'MSFT': MSFT_log_ret_close, 'PG': PG_log_ret_close}


# print(d_ret_tup)

df_ret = pd.DataFrame(data=d_ret, index=ret_dates)





### calculate portfoltio return ###
num_stocks = 20 + 10 +10 +10 + 4 + 5
VZ_port = 20/ num_stocks
INTC_port = 10/ num_stocks
JPM_port = 10/ num_stocks
AAPL_port = 10/ num_stocks
MSFT_port = 4/ num_stocks
PG_port = -5/ num_stocks

VZ_ret_cal = VZ_port*VZ_log_ret_close
INTC_ret_cal = INTC_port*INTC_log_ret_close
JPM_ret_cal = JPM_port*JPM_log_ret_close
AAPL_ret_cal = AAPL_port*AAPL_log_ret_close
MSFT_ret_cal = MSFT_port*MSFT_log_ret_close
PG_ret_cal = PG_port*PG_log_ret_close



# VZ_ret_cal = VZ_port*np.exp(VZ_log_ret_close)
# INTC_ret_cal = INTC_port*np.exp(INTC_log_ret_close)
# JPM_ret_cal = JPM_port*np.exp(JPM_log_ret_close)
# AAPL_ret_cal = AAPL_port*np.exp(AAPL_log_ret_close)
# MSFT_ret_cal = MSFT_port*np.exp(MSFT_log_ret_close)
# PG_ret_cal = PG_port*np.exp(PG_log_ret_close)

# port_return = np.log( VZ_ret_cal + INTC_ret_cal + JPM_ret_cal + AAPL_ret_cal + MSFT_ret_cal +PG_ret_cal )

# port_return = VZ_ret_cal + INTC_ret_cal + JPM_ret_cal + AAPL_ret_cal + MSFT_ret_cal + PG_ret_cal

port_return = np.diff(np.log(current_port_V))


#   JPM on 27/07/2016 is quoted NULL, averages of previous and next days were taken in the csv file



port_return_pd = pd.Series(port_return)



df_ret = df_ret.assign(port_return_pd = port_return_pd.values)

df_ret.to_csv('ret.csv', encoding='utf-8', index=True)







##### 1.1 Gauss pdf, QQ Plot, ACF, ACF sq. return , Variance Ratio Test ######

def Ret_Analysis(ret):

    ## Histogram with normal fit
    (mu, std) = norm.fit(port_return)
    n, bins, patches = plt.hist(port_return, 60, normed=1, facecolor='b')
    y = mlab.normpdf( bins, mu, std)
    l = plt.plot(bins, y, 'r--', linewidth=2)

    ## Moments Analysis
    deb = stats.describe(ret)

    plt.xlabel('Returns')
    plt.title('Histogram of Portfolio Return')
    plt.grid(True)
    plt.show()

    ## QQ plot
    stats.probplot(ret, dist="norm", plot=pylab)
    pylab.show()

    ## ACF plot
    plot_acf(ret)
    plt.title('ACF of Portfolio Return')
    plt.show()

    ## ACF plot of squared ret
    plot_acf(ret**2)
    plt.title('ACF of Squared Portfolio Return')
    plt.show()

    print(deb)

### VarRatio_test

def Variance_Ratio_Test(a, n):
    var_one_period = np.var(a)
    ret_nth_period = a[0::n]
    var_nth_period =np.var(ret_nth_period)
    var_ratio = var_nth_period/(n*var_one_period)
    z_var_ratio = (var_ratio - 1) / np.sqrt((2*(n-1))/(len(a)*n))
    print(z_var_ratio)

    return z_var_ratio




###### 1.2 Gauss VaR, Bootstrap HS VaR, Expected shortfall ######

def Gauss_VaR(ret, alpha,n): # roll_window %days
    daily_gauss_std = []
    gauss_daily_ret_VaR = []
    z_score = []
    ES_gauss_ret_VaR= []
    roll_window_pnl = []
    for i in range(0,df_ret.index.get_loc('2018-02-09')-df_ret.index.get_loc('2014-07-01')+1):

        roll_window_ret = ret[df_ret.index.get_loc('2014-07-01')-250+i:df_ret.index.get_loc('2014-07-01')+i]

        # first VaR estimates on 2014-07-01 for 2014-07-02, up to estimating VaR for 2018-02-10

        w = (np.mean(roll_window_ret))      # mean of the rolling window returns
        e = (np.std(roll_window_ret))       # standard deviation of the rolling window returns
        gauss_daily_ret_VaR.append(-(w*n -e*np.sqrt(n)*norm.ppf(alpha)))     # populating a list of rolling VaRs

        today_Gauss_ret_ES = -(w*n - e*np.sqrt(n)*norm.pdf(norm.ppf(alpha))/(1-alpha))
        # populating a list of rolling Expected SHortfall (ES)
        ES_gauss_ret_VaR.append(today_Gauss_ret_ES)
    return gauss_daily_ret_VaR, ES_gauss_ret_VaR

def HS_VaR(ret, alpha, n): # roll_window %days
    HS_daily_ret_VaR = []
    HS_ret_ES = []

    for i in range(0,df_ret.index.get_loc('2018-02-09')-df_ret.index.get_loc('2014-07-01')+1):
        sim = []
        roll_window_ret_HS = ret[df_ret.index.get_loc('2014-07-01') - 250 + i:df_ret.index.get_loc('2014-07-01') + i]
            # rolling_window, lookback of 250 days
        #for loop way   Uncomment to start
        for B in range(5000):
            # numpy randomly resampling nth days returns x T period, 250 days with replacement
            T_n_matrix = np.random.choice(roll_window_ret_HS, size = [250,n], replace = True)
            # sums up to n-period returns, returning T = 250, n-period returns
            sum_Tn = np.sum(T_n_matrix, axis = 1)
            # takes the 1 - alpha percentile of the returns
            sim.append(np.percentile(sum_Tn, (1- alpha)*100))
            sim_ES = sum_Tn[sum_Tn < np.percentile(sum_Tn, (1- alpha)*100)]
        #taking the mean of sim gives one forecast VaR, repeat to complete rolling estimates

        HS_daily_ret_VaR.append(-np.mean(sim))
        HS_ret_ES.append(-np.mean(sim_ES))

    return HS_daily_ret_VaR, HS_ret_ES




################    1.3/1.4 Kupiec Test    ################



def Kupiec_Test(VaR, alpha,test_alpha):

    # first VaR estimates on 2014-07-01 for 2014-07-02, up to estimating VaR for 2018-02-10

    sum_unVaR = 0

    # backtest return starts on 2014-07-02,ends on 2018-02-09

    backtest_port_return = df_ret['port_return_pd'].loc['2014-07-02':'2018-02-09']

    #backtest_port_return = port_return[1:(df_ret.index.get_loc('2018-02-09') - df_ret.index.get_loc('2014-07-01'))]


    VaR_less = VaR[0][0:-1]
    backtest_bool = np.less(backtest_port_return, np.multiply(VaR_less,-1))     # backtest return boolean array conditioning if exceeding VaR or not
    bool_unVaR = backtest_bool.astype(int)                                      # backtest return 1,0 binary array
    sum_unVaR = np.sum(bool_unVaR)                                              #  sums no. of exceeding VaR of 1,0



    Chi1 = stats.chi2.ppf(test_alpha, 1)

    d = len(backtest_port_return)
    j = sum_unVaR
    jd = j/d

    gauss_Kupiec_stat = -2*(j*np.log((1 - alpha)/jd) + (d - j)*np.log(alpha/(1 - jd)))

    VaR_less_ES = VaR[1][0:-1]

    # print(backtest_port_return)
    # print(VaR_less_ES)
    # print(VaR[1])

    backtest_bool = np.less(backtest_port_return, np.multiply(VaR_less_ES, -1))  # backtest return boolean array
    bool_ES = backtest_bool.astype(int)
    sum_ES = np.sum(bool_ES)      # backtest convert to 0,1 from boolean, which sums to no. of exceeding ES


    e = len(backtest_port_return)
    k = sum_ES
    ke = k/e

    ES_Kupiec_stat = -2*(k*np.log((1 - alpha)/ke) + (e - k)*np.log(alpha/(1 - ke)))

    # print(Chi1, sum_unVaR, gauss_Kupiec_stat, sum_ES, ES_Kupiec_stat, bool_unVaR, bool_ES)
    return Chi1, sum_unVaR, gauss_Kupiec_stat, sum_ES, ES_Kupiec_stat, bool_unVaR, bool_ES



################    1.5 backtest VaR graphs    ################

def backtest_VaR(VaR):
    backtest_port_return = df_ret['port_return_pd'].loc['2014-07-02':'2018-02-09']

    days = range(0, len(backtest_port_return))

    plot_VaR = np.multiply(VaR[0][0:-1], -1)        #   unconditional VaR plot
    plot_ES = np.multiply(VaR[1][0:-1], -1)         #   Expected Shortfall plot

    axes = plt.gca()
    axes.set_ylim([-0.1, 0.1])

    plt.plot(days, backtest_port_return, label = 'Portfolio Return', lw = '0.8')
    plt.plot(days, plot_VaR, label = 'unconditional VaR')
    plt.plot(days, plot_ES, label = 'Expected Shortfall')
    for i in range(0,len(backtest_port_return)):
        if backtest_port_return[i] < -VaR[0][i-1]:
            plt.axvline(x = i, c = 'g', lw = '0.5',ls = '--')

    for i in range(0,len(backtest_port_return)):
        if backtest_port_return[i] < -VaR[1][i-1]:
            plt.axvline(x = i, c = 'r', lw = '0.5', ls = '-')

    plt.xlabel('backtest time (day)')
    plt.legend()
    plt.title('VaR and ES backtesting')
    plt.show()


def Cond_Cov_Test(VaR, kupiec, test_alpha):

    # unpack kupiec statistics
    kupiec_VaR = kupiec[2]  # the uncoditional Likilihood Ratio of VaR
    kupiec_ES = kupiec[4]   # the uncoditional Likilihood Ratio of ES
    unVaR_bool = kupiec[5]  # boolean array of exceeding VaR
    ES_bool = kupiec[6]     # boolean array of exceeding ES


    tseries = range(0,len(unVaR_bool))

        # we sum the occurances of 1 to 1, 1 to 0, 0 to 1 and 0 to 0
        # of the boolean of array where 1 represents exceeding VaR and 0 otherwise

    t11 = sum(1 for i in tseries if unVaR_bool[i] == 1 and unVaR_bool[i-1] == 1
              and i!= 0)
    t10 = sum(1 for i in tseries if unVaR_bool[i] == 1 and unVaR_bool[i-1] == 0
              and i!= 0)
    t01 = sum(1 for i in tseries if unVaR_bool[i] == 0 and unVaR_bool[i-1] == 1
              and i!= 0)
    t00 = sum(1 for i in tseries if unVaR_bool[i] == 0 and unVaR_bool[i-1] == 0
              and i!= 0)

    t11_ES = sum(1 for i in tseries if ES_bool[i] == 1 and ES_bool[i - 1] == 1
              and i != 0)
    t10_ES = sum(1 for i in tseries if ES_bool[i] == 1 and ES_bool[i - 1] == 0
              and i != 0)
    t01_ES = sum(1 for i in tseries if ES_bool[i] == 0 and ES_bool[i - 1] == 1
              and i != 0)
    t00_ES = sum(1 for i in tseries if ES_bool[i] == 0 and ES_bool[i - 1] == 0
              and i != 0)

    # we calculate the first order Markov process probability matrix, Pi

    p01 = t01/ (t00+t01)
    p11 = t11/ (t10+t11)
    p00 = 1 - p01
    p10 = 1 - p11

    p01_ES = t01_ES / (t00_ES + t01_ES)
    p11_ES = t11_ES / (t10_ES + t11_ES)
    p00_ES = 1 - p01_ES
    p10_ES = 1 - p11_ES


    T = np.array([[t00,t01],[t10,t11]])     # occurrence matrix
    Pi = np.array([[p00,p01],[p10,p11]])    # probability transition matrix

    T_ES = np.array([[t00_ES, t01_ES], [t10_ES, t11_ES]])  # occurrence matrix of ES
    Pi_ES= np.array([[p00_ES, p01_ES], [p10_ES, p11_ES]])  # probability transition matrix of ES

    L_Pi = kupiec_VaR
    L_Pi_1 = (p00**t00)*(p01**t01)*(p10*t10)*(p11*t11)

    L_Pi_ES = kupiec_ES
    L_Pi_1_ES = (p00_ES ** t00_ES) * (p01_ES ** t01_ES) * (p10_ES * t10_ES) * (p11_ES * t11_ES)

    Chi1 = stats.chi2.ppf(test_alpha, 1)

    LR_ind = -2*np.log(L_Pi/L_Pi_1)         # independent Likelihood Ratio

    LR_cc = L_Pi + LR_ind

    LR_ind_ES = -2 * np.log(L_Pi_ES / L_Pi_1_ES)  # independent Likelihood Ratio

    LR_cc_ES = L_Pi_ES + LR_ind_ES

    Chi2 = stats.chi2.ppf(test_alpha, 2)


    print(Pi,T)
    print(Chi1, LR_ind, LR_cc, Chi2, LR_ind_ES, LR_cc_ES)
    return Chi1, LR_ind, LR_cc, Chi2, LR_ind_ES, LR_cc_ES



def Basel_Reg(VaR):
    basel_60_d_VaR = VaR[0][-60:]    # unpack last 60 VaRs, based on n-period returns
    basel_last_d_VaR = VaR[0][-1]
    MRC = max(basel_last_d_VaR, np.mean(basel_60_d_VaR))
    MRC_Cap = MRC*current_port_V[-1]
    print(current_port_V)
    print(MRC, MRC_Cap)
    return MRC


def HS_Bootstrap_Full_Val(df_ret,alpha, n):      # TODO historical simulation with full revaluation
    HS_daily_ret_VaR = []
    HS_ret_ES = []
    last_day_port_v = dg[dg.index.get_loc('2018-02-09'):]        ### test!

    for B in range(10000):
        # numpy randomly resampling nth days returns x T period, 250 days with replacement
        T_n_matrix = np.random.choice(df_ret, size=[250, n], replace=True)
        # sums up to n-period returns, returning T = 250, n-period returns
        sum_Tn = np.sum(T_n_matrix, axis=1)
        # calculate simulated stock price
        sim_stock_price = np.multiply(sum_Tn, last_day_port_v)          ##### test!!
        # takes the 1 - alpha percentile of the returns
        sim.append(np.percentile(sum_Tn, (1 - alpha) * 100))
        sim_ES = sum_Tn[sum_Tn < np.percentile(sum_Tn, (1 - alpha) * 100)]
    # taking the mean of sim gives one forecast VaR, repeat to complete rolling estimates

    HS_daily_ret_VaR.append(-np.mean(sim))
    HS_ret_ES.append(-np.mean(sim_ES))

    # print(sim)
    # print(HS_daily_ret_VaR)
    # print(HS_ret_ES)
    return HS_daily_ret_VaR, HS_ret_ES


# Black Scholes Option Price
def OptionPrice(S,K,sig,r,dt):
    d1 = np.divide((np.log(np.divide(S, K)) + ((r + (sig**2)/2)*dt)), sig*np.sqrt(dt))
    d2 = d1 - sig*np.sqrt(dt)
    c = np.multiply(S, norm.cdf(d1)) - np.multiply(K*np.exp(-r*dt), norm.cdf(d2))
    p = np.multiply(K*np.exp(-r*dt), norm.cdf(-d2)) - np.multiply(S, norm.cdf(-d1))
    return c, p

def Greeks(S0, K, r, sig, dt,q):
    d1 = np.divide((np.log(np.divide(S0, K)) + ((r + (sig ** 2) / 2) * dt)), sig * np.sqrt(dt))
    d2 = d1 - sig * np.sqrt(dt)
    call_delta = norm.cdf(d1)*np.exp(-q*dt)
    put_delta = (norm.cdf(d1) - 1)*np.exp(-q*dt)
    gamma = (1/(S0*sig*np.sqrt(2*np.pi*dt)))*np.exp((-d1**2)/2)*np.exp(-q*dt)
    call_theta = (1/360)*(-(gamma*(S0**2)*(sig**2)/2) - r*K*np.exp(-r*dt)*norm.cdf(d2) + q*S0*np.exp(-q*dt)*norm.cdf(d1))
    put_theta = (1/360)*(-(gamma*(S0**2)*(sig**2)/2) + r*K*np.exp(-r*dt)*norm.cdf(-d2) - q*S0*np.exp(-q*dt)*norm.cdf(-d1))
    print(call_theta, put_theta, call_delta, put_delta, gamma)
    return call_theta, put_theta, call_delta, put_delta, gamma

 ####   HS_Bootstrap_Full_Reval          #TODO historical simulation with full revaluation

def HS_Bootstrap_Full_Val(alpha, epsilon, n, B):

    # matrix of the 6 stocks returns of 1536 days
    d_ret_tup = np.transpose(np.array([VZ_log_ret_close,INTC_log_ret_close, JPM_log_ret_close , AAPL_log_ret_close,  MSFT_log_ret_close, PG_log_ret_close]))


    # matrix of randomly sample B times of the six returns, n days

    T_n_matrix = d_ret_tup[np.random.choice(d_ret_tup.shape[0], size=[B,n], replace=True)]

    # convert daily return to n-period returns

    sum_Tn = np.sum(T_n_matrix, axis = 1)

    # calculate bootstrap stock values by multiplying boststrap stock returns and last day stock value
    # print(np.exp(sum_Tn))
    # print(last_day_port_v)
    ten_day_port_value = np.multiply(np.exp(sum_Tn), last_day_port_v)
    # print(ten_day_port_value)

    # stock order in , 1. VZ    2. INTC     3. JPM      4. AAPL     5. MSFT     6. PG

    ###     initial option values on 2018-02-09      ####

    call_int_AAPL= 20.05    # short 7, 140 calls, Apr 20, implied vol = 0.3954
    put_int_MSFT = 12.225   # long 6, 95 puts, Jul 20, implied vol = 0.4103
    call_int_PG = 1.695      # long 10, 85 calls, Jun 16, implied vol = 0.2046

    opt_int = np.array([call_int_AAPL, put_int_MSFT, call_int_PG])


    ###     maturity of the options     ###
    call_aapl_days = date(2018,2,9) - date(2018,4,20)       # 70
    call_aapl_days = - call_aapl_days.days
    put_msft_days = date(2018,2,9) - date(2018,7,20)        # 161
    put_msft_days = - put_msft_days.days
    call_pg_days = date(2018,2,9) - date(2018,6,15)         # 126
    call_pg_days = - call_pg_days.days

    ###     risk free rate of the options   ###
    rate_aapl = 0.0168324 + (call_aapl_days-60)*(0.0181050-0.0168324)/(90-60)
    rate_msft = 0.0181050 + (put_msft_days-90)*(0.0202633-.0181050)/(180-90)
    rate_pg = 0.0181050 + (call_pg_days-90)*(0.0202633-.0181050)/(180-90)

    # calculate AAPL calls, MSFT puts, PG calls, 10 day simulated prices

    aapl_opt = OptionPrice(ten_day_port_value[:,3], 140, 0.3954, rate_aapl, call_aapl_days/360)[0]
    msft_opt = OptionPrice(ten_day_port_value[:,4], 95, 0.4103, rate_msft, put_msft_days/360)[1]
    pg_opt = OptionPrice(ten_day_port_value[:,5], 85, 0.2046, rate_pg, call_pg_days/360)[0]

    # Simulated Option portfolio value, 1. AAPL calls 2. MSFT puts 3. PG Calls
    opt_ten_val = np.array([aapl_opt, msft_opt, pg_opt])
    opt_ten_vel = np.transpose(opt_ten_val)

    # Simulated Full portfolio value
    ten_day_port_value_w_opt = np.concatenate((ten_day_port_value, opt_ten_vel), axis = 1)


    # multiply the ten day portfolio value by number of holdings/ weights
    weights = np.array([20, 10, 10, 10, 4, -5, -7, 6, 10])
    weighted_ten_day_port_w_opt = np.multiply(ten_day_port_value_w_opt, weights)

    # Getting simulated portfolio returns, including option returns
    # First add initial option values to the list of securities prices on 2018-02-09
    last_day_port_w_opt = np.append(last_day_port_v, opt_int)
    weighted_last_day_port_w_opt = np.sum(np.multiply(last_day_port_w_opt, weights), axis =0)

    # Calculate simulated portfolio returns by dividing simulated prices by the security prices on 2018-02-09 respectively
    ten_day_sim_ret = np.log(np.divide(np.sum(weighted_ten_day_port_w_opt, axis = 1), weighted_last_day_port_w_opt))
    plt.hist(ten_day_sim_ret)
    plt.show()

    # Calculate simulated total portfolio returns

    # Calculate VaR by locating the 1st percentile
    HS_full_Val = -np.percentile(ten_day_sim_ret, (1 - alpha)*100)
    print(HS_full_Val)


    ############     Calculate Marginal VaR and Conditional VaR of HS Full Revaluation    #########


    ten_day_sim_port_ret_HS = ten_day_sim_ret   # simulated portfolio returns
    HS_full_Val_VaR = HS_full_Val               # calculated VaR
    ten_day_sim_sec_ret_HS = np.log(np.divide(ten_day_port_value_w_opt, last_day_port_w_opt)) # calculate simulated securities return

    l = []

    # Conditioning on portfolio returns between -VaR - epsilon and -VaR + epsilon, return list of conditioned simulated securities returns
    for i in range(0,len(ten_day_sim_port_ret_HS)):
        if (-HS_full_Val_VaR - epsilon) <= ten_day_sim_port_ret_HS[i] <= (-HS_full_Val_VaR + epsilon):
            l.append(ten_day_sim_sec_ret_HS[i])
    l = np.asarray(l)


    # Calculate MVaR by taking the mean of each conditioned securities return
    MVaR = -np.mean(l, axis = 0 )


    # cauclate percentage weights

    sum_weights = np.sum(np.array([20, 10, 10, 10, 4, 5, 7, 6, 10]))
    percent_weights = np.divide(np.array([20, 10, 10, 10, 4, 5, 7, 6, 10]), sum_weights)

    CVaR = np.multiply(MVaR, percent_weights)
    print(np.sum(CVaR))

    name = np.array(["VZ","INTC","JPM","AAPL","MSFT","PG","AAPL opt", "MSFT opt","PG opt"])

    print(pd.DataFrame(MVaR, CVaR))

    return MVaR, CVaR

###########  HS Boostrap with Delta and Gamma       #######TODO Historical Simualtion with Greeks Approximation

def HS_Bootstrap_Greek_App(alpha, epsilon_greeks, n, B):


    # calculate Greeks of AAPL calls, MSFT puts, PG calls

    ###     maturity of the options     ###
    call_aapl_days = date(2018, 2, 9) - date(2018, 4, 20)  # 70
    call_aapl_days = - call_aapl_days.days
    put_msft_days = date(2018, 2, 9) - date(2018, 7, 20)  # 161
    put_msft_days = - put_msft_days.days
    call_pg_days = date(2018, 2, 9) - date(2018, 6, 15)  # 126
    call_pg_days = - call_pg_days.days

    ###     risk free rate of the options   ###
    rate_aapl = 0.0168324 + (call_aapl_days - 60) * (0.0181050 - 0.0168324) / (90 - 60)
    rate_msft = 0.0181050 + (put_msft_days - 90) * (0.0202633 - .0181050) / (180 - 90)
    rate_pg = 0.0181050 + (call_pg_days - 90) * (0.0202633 - .0181050) / (180 - 90)

    aapl_greeks = Greeks(last_day_port_v[:, 3], 140, rate_aapl, 0.3954, call_aapl_days / 360, 0)
    msft_greeks = Greeks(last_day_port_v[:, 4], 95, rate_msft, 0.4103, put_msft_days / 360, 0)
    pg_greeks = Greeks(last_day_port_v[:, 5], 85, rate_pg, 0.2046, call_pg_days / 360, 0)

    # matrix of the 6 stocks returns of 1536 days
    d_ret_tup = np.transpose(np.array(
        [VZ_log_ret_close, INTC_log_ret_close, JPM_log_ret_close, AAPL_log_ret_close, MSFT_log_ret_close,
         PG_log_ret_close]))

    # Bootstrap stock prices

    # matrix of randomly sample 1000 times of the six returns, n days

    T_n_matrix_greeks = d_ret_tup[np.random.choice(d_ret_tup.shape[0], size=[B, n], replace=True)]

    # convert daily return to n-period returns

    sum_Tn_greeks = np.sum(T_n_matrix_greeks, axis=1)

    # calculate bootstrap stock values by multiplying bootstrap stock returns and last day stock value

    ten_day_port_value_greeks = np.multiply(np.exp(sum_Tn_greeks), last_day_port_v)

    ######## We now caculate simulated option prices using simulated stock prices and option greeks

    # stock ordering, 1. VZ    2. INTC     3. JPM      4. AAPL     5. MSFT     6. PG
    # Greeks ordering 1. call_theta     2. put_theta    3. call_delta   4. put_delta    5. gamma

    # calculate change in stock prices
    dS = ten_day_port_value_greeks - last_day_port_v

    # calculate new option prices

    call_int_AAPL = 20.05  # short 7, 140 calls, Apr 20, implied vol = 0.3954
    put_int_MSFT = 12.225  # long 6, 95 puts, Jul 20, implied vol = 0.4103
    call_int_PG = 1.695  # long 10, 85 calls, Jun 16, implied vol = 0.2046

    opt_int = np.array([call_int_AAPL, put_int_MSFT, call_int_PG])

    aapl_opt_greeks_val = call_int_AAPL + np.multiply(aapl_greeks[0], 10 / 360) + np.multiply(aapl_greeks[2],
                                                                                              dS[:, 3]) + np.multiply(
        0.5 * aapl_greeks[4], np.power(dS[:, 3], 2))
    msft_opt_greeks_val = put_int_MSFT + np.multiply(msft_greeks[1], 10 / 360) + np.multiply(msft_greeks[3],
                                                                                             dS[:, 4]) + np.multiply(
        0.5 * msft_greeks[4], np.power(dS[:, 4], 2))
    pg_opt_greeks_val = call_int_PG + np.multiply(pg_greeks[0], 10 / 360) + np.multiply(pg_greeks[2],
                                                                                        dS[:, 5]) + np.multiply(
        0.5 * pg_greeks[4], np.power(dS[:, 5], 2))

    opt_ten_val_greeks = np.array([aapl_opt_greeks_val, msft_opt_greeks_val, pg_opt_greeks_val])
    opt_ten_vel_greeks = np.transpose(opt_ten_val_greeks)

    # Simulated Full portfolio value, we combine stock prices and option prices
    ten_day_port_value_w_opt_greeks = np.concatenate((ten_day_port_value_greeks, opt_ten_vel_greeks), axis=1)

    # multiply the ten day portfolio value by number of holdings/ weights
    weights = np.array([20, 10, 10, 10, 4, -5, -7, 6, 10])
    weighted_ten_day_port_w_opt_greeks = np.multiply(ten_day_port_value_w_opt_greeks, weights)

    # Getting simulated portfolio returns, including option returns
    # First adding initial option values to the list of securities prices on 2018-02-09
    last_day_port_w_opt_greeks = np.append(last_day_port_v, opt_int)
    weighted_last_day_port_w_opt_greeks = np.sum(np.multiply(last_day_port_w_opt_greeks, weights), axis=0)

    # Second Calculate simulated portfolio returns by dividing simulated prices by the security prices on 2018-02-09 respectively
    ten_day_sim_ret_greeks = np.log(
        np.divide(np.sum(weighted_ten_day_port_w_opt_greeks, axis=1), weighted_last_day_port_w_opt_greeks))
    plt.hist(ten_day_sim_ret_greeks)
    # plt.show()

    # Calculate VaR by locating the 1st percentile
    HS_Val_greeks = -np.percentile(ten_day_sim_ret_greeks, (1 - alpha) * 100)
    # print(HS_Val_greeks)

    ############     Calculate Marginal VaR and Conditional VaR of HS Delta and Gamma Approximation    #########

    ten_day_sim_port_ret_HS_greeks = ten_day_sim_ret_greeks  # simulated portfolio returns
    HS_Val_VaR_greeks = HS_Val_greeks  # calculated VaR
    ten_day_sim_sec_ret_HS_greeks = np.log(np.divide(ten_day_port_value_w_opt_greeks, last_day_port_w_opt_greeks))  # calculate simulated securities return

    l_greeks = []

    # Conditioning on portfolio returns between -VaR - epsilon and -VaR + epsilon, return list of conditioned simulated securities returns
    for i in range(0, len(ten_day_sim_port_ret_HS_greeks)):
        if (-HS_Val_VaR_greeks - epsilon_greeks) <= ten_day_sim_port_ret_HS_greeks[i] <= (
                -HS_Val_VaR_greeks + epsilon_greeks):
            l_greeks.append(ten_day_sim_sec_ret_HS_greeks[i])
    l_greeks = np.asarray(l_greeks)
    # l_greeks[:,6] = l_greeks[~np.isnan(l_greeks[:,6])]

    # print(np.count_nonzero(~np.isnan(l_greeks[:,6])))
    # print((l_greeks[:,6]))
    print(HS_Val_VaR_greeks)
    # print(np.mean(l_greeks[:,6]))


    # Calculate MVaR by taking the mean of each conditioned securities return
    MVaR_greeks = -np.mean(l_greeks, axis=0)

    # cauclate percentage weights

    sum_weights_greeks = np.sum(np.array([20, 10, 10, 10, 4, 5, 7, 6, 10]))
    percent_weights_greeks = np.divide(np.array([20, 10, 10, 10, 4, 5, 7, 6, 10]), sum_weights_greeks)

    CVaR_greeks = np.multiply(MVaR_greeks, percent_weights_greeks)
    print(np.sum(CVaR_greeks))
    print(pd.DataFrame(MVaR_greeks, CVaR_greeks))
    return MVaR_greeks, CVaR_greeks


def main():

    start_time = time.time()

    # Ret_Analysis(port_return)
    # Variance_Ratio_Test(port_return, 10000)

    ## 1 day period return

    # Gauss_VaR(port_return,0.99,1)
    # Kupiec_Test(Gauss_VaR(port_return,0.99,1), 0.99, 0.95)
    # backtest_VaR(Gauss_VaR(port_return,0.99,1))
    # Cond_Cov_Test(Gauss_VaR(port_return,0.99,1), Kupiec_Test(Gauss_VaR(port_return,0.99,1), 0.99, 0.95),0.95)
    #
    # Gauss_VaR(port_return, 0.90,1)
    # Kupiec_Test(Gauss_VaR(port_return, 0.90,1), 0.90, 0.95)
    # backtest_VaR(Gauss_VaR(port_return, 0.90,1))
    # Cond_Cov_Test(Gauss_VaR(port_return,0.90,1), Kupiec_Test(Gauss_VaR(port_return,0.90,1), 0.99, 0.95),0.95)

    ## 10 days period return

    # HS_VaR(port_return, 0.90,10)        ## ~10 mins runtime(n =10, B = 5000)
    # Kupiec_Test(HS_VaR(port_return, 0.90, 10), 0.90, 0.95)
    # backtest_VaR(HS_VaR(port_return, 0.90,10))
    # Cond_Cov_Test(HS_VaR(port_return, 0.90, 10), Kupiec_Test(HS_VaR(port_return,0.90,1), 0.99, 0.95),0.95)

    # HS_VaR(port_return, 0.99,10)
    # Kupiec_Test(HS_VaR(port_return, 0.99,10), 0.99, 0.95)
    # backtest_VaR(HS_VaR(port_return, 0.99,10))
    # Cond_Cov_Test(HS_VaR(port_return, 0.99,10), Kupiec_Test(HS_VaR(port_return,0.99,10), 0.99, 0.95),0.95)

    ## Basel regulated MRC

    # Basel_Reg(Gauss_VaR(port_return,0.99,10))
    # Basel_Reg(HS_VaR(port_return, 0.99, 10))

    ## Bootstrap with Options

    HS_Bootstrap_Full_Val(0.99, 0.005, 10, 100000)
    # HS_Bootstrap_Greek_App(0.95, 0.005, 10, 1000000)

    print("--- %s seconds ---" % (time.time() - start_time))


main()






#####################################       End         ############################################################
