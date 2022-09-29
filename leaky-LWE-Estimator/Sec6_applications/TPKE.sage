load("../framework/instance_gen.sage")

# == Kyber Parameter == 
Kyber512 = (256,2,3329,3,2)
Kyber768 = (256,3,3329,2,2)
Kyber1024 = (256,4,3329,2,2)

# == TPKE Kyber Parameter == 
q_scale_factor = 4
TKyber512 = (256,2,q_scale_factor * 3329,3,2)
TKyber768 = (256,3,q_scale_factor * 3329,2,2)
TKyber1024 = (256,4,q_scale_factor * 3329,2,2)


def concrete_security(kyber_param,l,sigma,tailcut):
	(d,k,q,eta1,eta2)=kyber_param #parse parameters
	n=d*k #LWE dimension = ring degree * module rank
	m=n #quadratic matrix
	D_s = build_centered_binomial_law(eta1) #LWE secret distribution
	D_e = D_s #LWE noise distribution
	A,b,dbdd = initialize_from_LWE_instance(DBDD_predict,n,q,m,D_e,D_s) 
	beta, delta = dbdd.estimate_attack() #starting security level (without hints)
	
	D_r = build_centered_binomial_law(eta1) # ciphertext secret distribution
	D_f = build_centered_binomial_law(eta2) # ciphertext noise distribution
	D_n = build_Gaussian_law(sigma,tailcut*sigma) # flooding noise distribution (= sigma_real in security_estimates python script)
	var = sigma^2 
	for i in range(l*d):
		v = vec([draw_from_distribution(D_r) for i in range(n)] + [draw_from_distribution(D_f) for j in range(m)])
		_ = dbdd.integrate_approx_hint(v,dbdd.leak(v)+draw_from_distribution(D_n),var,aposteriori=False)
		if (i%20==0):
			print("We are currently at hint",i)
	return "Slut"
	



