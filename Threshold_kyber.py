from math import log
from operator import indexOf
from Kyber_failure import *
from MLWE_security import MLWE_summarize_attacks, MLWEParameterSet
from proba_util import build_mod_switching_error_law, iter_law_convolution

class ThresholdPKEParameterSet:
    def __init__(self, n, d, m, ks, ke,  q, rqk, rqc, rq2, ke_ct=None):
        if ke_ct is None:
            ke_ct = ke
        self.n = n      # number of parties
        self.d = d      # polynomial ring degree
        self.m = m      # module rank r
        self.ks = ks     # binomial parameter for the secret key
        self.ke = ke    # binomial parameter for the pk error
        self.ke_ct = ke_ct    # binomial parameter for the ciphertext errors
        self.q = q
        # Truncation parameters: not using for now
        self.rqk = rqk  # 2^(bits in the public key)
        self.rqc = rqc  # 2^(bits in the first ciphertext)
        self.rq2 = rq2  # 2^(bits in the second ciphertext)


def estimate_security(ps):
    raise NotImplementedError

""" def Kyber_to_MLWE(kps):
    if kps.ks != kps.ke:
        raise "The security script does not handle different error parameter in secrets and errors (ks != ke) "

    # Check whether ciphertext error variance after rounding is larger than secret key error variance
    Rc = build_mod_switching_error_law(kps.q, kps.rqc)
    var_rounding = sum([i*i*Rc[i] for i in Rc.keys()])

    if kps.ke_ct/2. + var_rounding < kps.ke/2.:
        raise "The security of the ciphertext MLWE may not be stronger than the one of the public key MLWE"    

    return MLWEParameterSet(kps.d, kps.m, kps.m + 1, kps.ks, kps.q)
 """


def find_flooding_sigma(ps, max_failure=2**(-64), sigma_start=230, sigma_step=5, sd_tailcut=3.1):
    """ Find the largest flooding parameter that still gives correct threshold decryption """
    failure_prob = 1
    ct_error = p2_cyclotomic_final_error_distribution(ps) # initial ct error distribution
    sigma = sigma_start
    while failure_prob > max_failure:
        sigma -= sigma_step
        sigma_law = build_rounded_gaussian_law(sigma, tailcut=int(round(sd_tailcut*sigma)))
        # sum over n parties
        # TODO: this is only correct for full-threshold. Also really slow (for large q, should approximate instead)
        summed_sigmas = iter_law_convolution(sigma_law, ps.n)
        final_error_distribution = law_convolution(ct_error, summed_sigmas)
        failure_prob = ps.d * tail_probability(final_error_distribution, ps.q/4)
        approx_shift = int(ps.q/4 - ps.n * sd_tailcut * sigma)
        #print('failure:', failure_prob)
        if failure_prob < 2**(-200):
            print('\tTrying sigma: %s, log failure: -inf, shift ~ %.2f' % (sigma, approx_shift))
        else:
            print('\tTrying sigma: %s, log failure: %.2f, shift ~ %.2f' % (sigma, log(failure_prob, 2), approx_shift))

    pr_ct_max = max(abs(v) for v in ct_error if ct_error[v] > max_failure)
    pr_flood_max = max(abs(v) for v in final_error_distribution if final_error_distribution[v] > max_failure)
    print('\t[worst ct max: %s, max w/ pr. 2^{%.2f}: %s]' % (max(ct_error), log(ct_error[pr_ct_max],2), pr_ct_max))
    print('\t[flood max: 2^{%.2f}: %s, q/4: %s]' % (log(final_error_distribution[pr_flood_max],2), pr_flood_max, q/4))
    
    shift = pr_ct_max
    print('Found sigma: %s, shift: %s' % (sigma, shift))
    return sigma, shift

def find_best_rd(ps, sigma_real, shift, sd_tailcut=3.1, sigma_step=5):
    """ Given a flooding parameter and a shift, find Gaussian D_sim parameter with the smallest Renyi divergence
    
    Returns a list of Renyi divergences and corresponding list of orders from 2-10 """
    sigma_law = build_rounded_gaussian_law(sigma_real, tailcut=int(round(sd_tailcut*sigma_real)))
    D_real = law_add_constant(sigma_law, shift)
    # print('Real dist tails, left: %s, right: %s' % (-int(round(sd_tailcut*sigma)) + shift, int(round(sd_tailcut*sigma)) + shift))
    rd_old = 2**40
    rd_new = 2**40-1
    rds = []
    rds_old = []
    sigma_sim = sigma_real
    print('Finding best RD: sigma_real =', sigma_real)
    # Increase sigma until RD stops decreasing
    while rd_new < rd_old - 0.0001:
        try:
            D_sim = build_rounded_gaussian_law(sigma_sim, tailcut=int(round(sd_tailcut*sigma_real + shift)))
            tmp = rd_new
            rd_new = compute_rd(D_real, D_sim)
            rd_old = tmp
            rds_old = rds
            print('Sigma sim: %.2f, RD: %.10f' % (sigma_sim, rd_new))
            rds = [compute_rd(D_real, D_sim, i) for i in range(2,11)]
            #for i,rd in enumerate(rds):
            #    print('rd %s: %.2f, ' % (i+2, rds[i]), sep='')
            #print()
            sigma_sim += sigma_step
        except (ZeroDivisionError, KeyError):
            sigma_sim += sigma_step
            #continue
        #    #print('Trying ideal sigma:', sigma)
        #    continue
    print('RD: ', rd_old)
    print('rds:', rds_old)
    return sigma_sim - sigma_step, rds_old, range(2,11)

def compute_sec_from_rd(ps, base_rd, a, num_samples, share_size=None, max_T=None, base_sec=128):
    """ Calculate OW-CPA security level using RD(D_real, D_sim) and no. samples etc.
    
    Returns the best security level out of the given RDs of different orders """
    # assume additive secret sharing if unspecified
    if share_size is None:
        share_size = 1
    if max_T is None:
        max_T = ps.n - 1
    log_final_rd = [log(rd,2) * ps.d * num_samples * (ps.n*share_size - max_T) * (aa-1)/aa for (rd,aa) in zip(base_rd, a)]
    final_secs = [base_sec * (aa-1)/aa - rd for rd,aa in zip(log_final_rd, a)]
    print('log(final rd): ', final_secs)
    best_sec = max(final_secs)
    best_a = indexOf(final_secs, best_sec) + 2
    return best_sec, best_a
    #return base_sec * (a-1)/a - log_final_rd

def communication_costs(ps):
    """ Compute the communication cost of a parameter set
    :param ps: Parameter set (ParameterSet)
    :returns: (cost_Alice, cost_Bob) (in Bytes)

    TODO: adapt for threshold setting
    """
    A_space = 256 + ps.d * ps.m * log(ps.rqk)/log(2)
    B_space = ps.d * ps.m * log(ps.rqc)/log(2) + ps.d * log(ps.rq2)/log(2)
    return (int(round(A_space))/8., int(round(B_space))/8.)


def summarize(ps):
    print ("params: ", ps.__dict__)
    print ("com costs: ", communication_costs(ps))
    F, f = p2_cyclotomic_error_probability(ps)
    print ("failure: %.1f = 2^%.1f"%(f, log(f + 2.**(-300))/log(2)))

def check_rd(sigma_real, sigma_sim, shift_max, a, sd_tailcut, target_rd):
    """ Sanity check that RD_a(D_real + shift || D_sim) < target_rd for all shifts smaller than shift_max """
    D_real = build_rounded_gaussian_law(sigma_real, tailcut=int(round(sd_tailcut*sigma_real)))
    D_sim = build_rounded_gaussian_law(sigma_sim, tailcut=int(round(sd_tailcut*sigma_real + shift_max)))

    for shift in range(0, shift_max, 5):
        D_real_shifted = law_add_constant(D_real, shift)
        rd = compute_rd(D_real_shifted, D_sim, a)
        if rd > target_rd:
            raise Error('RD too large! Shift %s, shift_max %s' % (shift, shift_max))
    print('RD check OK')

if __name__ == "__main__":
    q = 2*3329
    nparties = 2
    ps = ThresholdPKEParameterSet(nparties, 256, 4, 1, 1, q, 2**12, 2**12, 2**12)
    nsamples = 1
    gaussian_tailcut = 1.75

    sigma, shift = find_flooding_sigma(ps, sigma_step=20, sigma_start=q/12, sd_tailcut=gaussian_tailcut)
    sigma_sim, rd_list, a_list = find_best_rd(ps, sigma, shift, sigma_step=20, sd_tailcut=gaussian_tailcut)
    sec, a = compute_sec_from_rd(ps, rd_list, a_list, nsamples, base_sec=256)
    check_rd(sigma, sigma_sim, shift, a, gaussian_tailcut, rd_list[a-2])
    print('256-bit ----> %.2f-bit security, using RD_%s' % (sec, a))

    # Parameter sets
    # ps_light = KyberParameterSet(256, 2, 3, 3, 3329, 2**12, 2**10, 2**4, ke_ct=2)
    # ps_recommended = KyberParameterSet(256, 3, 2, 2, 3329, 2**12, 2**10, 2**4)
    # ps_paranoid = KyberParameterSet(256, 4, 2, 2, 3329, 2**12, 2**11, 2**5)

    # # Analyses
    # print ("Kyber512 (light):")
    # print ("--------------------")
    # print ("security:")
    # MLWE_summarize_attacks(Kyber_to_MLWE(ps_light))
    # summarize(ps_light)
    # print ()

    # print ("Kyber768 (recommended):")
    # print ("--------------------")
    # print ("security:")
    # MLWE_summarize_attacks(Kyber_to_MLWE(ps_recommended))
    # summarize(ps_recommended)
    # print ()

    # print ("Kyber1024 (paranoid):")
    # print ("--------------------")
    # print ("security:")
    # MLWE_summarize_attacks(Kyber_to_MLWE(ps_paranoid))
    # summarize(ps_paranoid)
    # print ()
