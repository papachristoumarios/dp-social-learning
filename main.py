import argparse
import numpy as np
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=100)
    parser.add_argument('--eps', type=float, default=1)
    parser.add_argument('-p', type=float, default=0.75)
    parser.add_argument('--method', type=str, default='private_signal', choices=['private_signal', 'local_dp', 'exponential_mechanism'])

    return parser.parse_args()

def get_actions(args, P_X, method='private_signal'):
    p = args.p
    n = args.n
    eps = args.eps
    rho_eps = 1 / (1 + np.exp(-eps))

    pi_sS = np.zeros(n) # holds the belief of the omniscient observer for theta = 1
    pi_sA = np.zeros(n) # holds the belief of a player for theta = 1

    Du = np.zeros(n) # sensitivity
    a = np.zeros(n) # actions
    s = np.zeros(n) # signals

    for i in range(n):
        u = int(np.random.uniform() >= 0.5)
        s[i] = int(np.random.uniform() >= P_X[1, u])

        pi_sS[i] = (s[:i] == 1).sum() / (i + 1)
        pi_sA[i] = ((s[i] == 1) + (a[:i - 1] == 1).sum()) / (i + 1)
        
        if method == 'private_signal':
            # Each agent sets a_t = s_t
            a[i] = s[i]
        elif method == 'local_dp':
            # Each agent applies local DP to their private signal and reports it,
            # i.e. z_t ~ Be(rho_eps) and if z_t = 1 then a_t is the flipped s_t
            z = int(np.random.uniform() >= rho_eps)
            a[i] = z * (1 - s[i]) + (1 - z) * s[i]
        elif method == 'exponential_mechanism':
            pi_1 = pi_sA[i]
            pi_0 = 1 - pi_1

            z_11 = (1 + (a[:i - 1] == 1).sum()) / (i + 1)
            z_10 = (a[:i-1] == 1).sum() / (i + 1)

            z_01 = (a[:i-1] == 0).sum() / (i + 1)
            z_00 = (1 + (a[:i - 1] == 0).sum()) / (i + 1)

            Du[i] = max(abs(z_11 - z_10), abs(z_01 - z_00))

            num_1 = np.exp(eps * pi_1 / (2 * Du[i]))
            num_2 = np.exp(eps * pi_0 / (2 * Du[i]))

            den = num_1 + num_2

            p_1 = num_1 / den
            p_2 = num_2 / den

            a[i] = int(np.random.uniform() >= p_1)

    lbr_sS = np.log(pi_sS / (1 - pi_sS))
    lbr_sA = np.log(pi_sA / (1 - pi_sA))

    return a, lbr_sS, lbr_sA, pi_sS, pi_sA, Du

def kl_divergence_bern(p, q):
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

if __name__ == '__main__':
    args = get_args()
    p = args.p
    n_range = 1 + np.arange(args.n)
   
    # State of the world
    # (future) add different priors 
    theta = int(np.random.uniform() >= 0.5)
   
    # Private signal probabilities P_X[i, j] corresponds to Pr[X = i | theta = j]
    P_X = np.array([[p, 1 - p], [1 - p, p]])

    a_signal, lbr_sS, lbr_sA, pi_sS, pi_sA, Du = get_actions(args, P_X, method=args.method)
    a_correct_signal = (a_signal == theta).astype(np.float64)
    p_correct_signal = np.cumsum(a_correct_signal) / n_range

    if args.method == 'exponential_mechanism':
        num_plots = 5
    else:
        num_plots = 4

    fig, ax = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))

    ax[0].set_title('Actions')
    ax[0].set_xlabel('Agent $t$')

    ax[0].plot(a_signal, label=args.method)
    ax[0].legend()

    ax[1].set_title('LBR_t / t')
    ax[1].set_xlabel('Agent $t$')
    
    ax[1].plot(lbr_sA / n_range, label='LBR of player')
    ax[1].plot(lbr_sS / n_range, label='LBR of omniscient observer')

    ax[1].legend()

    ax[2].set_title('Probability of Taking the Correct Action')
    ax[2].set_xlabel('Agent $t$')
    ax[2].plot(p_correct_signal, label=args.method)

    ax[2].legend()

    ax[3].set_title('Belief for $\\theta = 1$')
    ax[3].set_xlabel('Agent $t$')
    ax[3].plot(pi_sA, label='Belief of player')
    ax[3].plot(pi_sS, label='Belief of omniscient observer')
    ax[3].legend()

    if args.method == 'exponential_mechanism':
        ax[4].set_title('Sensitivity')
        ax[4].set_xlabel('Agent $t$')
        ax[4].plot(Du, label=args.method)
        ax[4].legend()

    plt.suptitle(f'$\\epsilon = {args.eps}, n = {args.n}, p = {args.p}$, method = {args.method}')

    plt.savefig('social_learning.pdf')

    print(kl_divergence_bern(args.p, 1 - args.p))

