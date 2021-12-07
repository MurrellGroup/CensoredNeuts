using Distributions, MCMCDiagnostics

#NOTE: -----------Assumes IC50s are all in log10 domain-------------
#------------------------------MCMC functions------------------------------
#=
random variables:
latent neut distribution (could ignore this)
change mean
change std
measurement std
latent neut values ~ latent neut distribution
changes ~ change distribution
when a neut value is below threshold, P(value) = integral over implied gaussian for everything below the cutoff.
=#
function MCMC(founder_neuts, variant_neuts, censor_at; prop_bw = 0.02, maxiters = 10000000) 
    
    log_censored_probability(mu,std) = logcdf(Normal(mu,std),censor_at)

    function point_log_likelihoods(points,mus,std; thresh = censor_at)
        LLs = logpdf(Normal(0.0,std),points .- mus)
        for i in 1:length(points)
            if points[i] <= thresh
                LLs[i] = log_censored_probability(mus[i],std)
            end
        end
        return sum(LLs)
    end

    function propose_some(v,std,prop)
        newv = copy(v)
        for i in 1:length(newv)
            if rand()<prop
                newv[i] = v[i] .+ randn()*std
            end
        end
        return newv
    end

    log_change_prior(mu,std) = logpdf(Normal(0.0,10.0),mu) + logpdf(Gamma(2.0,0.1),std)

    #Error STD expected to be fairly small.
    log_error_prior(error_std) = logpdf(Gamma(1.0,0.2),error_std)

    #Could intro hyperparams over this, but it won't make a diff.
    latent_neut_dist(latent_neuts) = sum(logpdf(Normal(2.5,5.0),latent_neuts))

    function log_lik(founder_neuts, variant_neuts, latent_neuts, changes, error_std)
        LL = point_log_likelihoods(founder_neuts,latent_neuts,error_std)
        LL += point_log_likelihoods(variant_neuts,latent_neuts .- changes,error_std)
        return LL
    end

    function log_prior(latent_neuts, changes, change_mean, change_std, error_std)
        LP = latent_neut_dist(latent_neuts)
        LP += sum(logpdf(Normal(change_mean,change_std),changes)) #P(changes|change dist)
        LP += log_change_prior(change_mean,change_std)
        LP += log_error_prior(error_std)
        return LP
    end
    
    
    #Slightly_guided_init
    error_std = 0.1
    change_mean = mean((founder_neuts .- variant_neuts)[variant_neuts .> censor_at])
    change_std = std((founder_neuts .- variant_neuts)[variant_neuts .> censor_at])
    latent_neuts = founder_neuts .+ randn()*0.1
    changes = (ones(N) .* change_mean) .+ randn(N)*0.3
    
    Lpost = log_lik(founder_neuts, variant_neuts, latent_neuts, changes, error_std) + log_prior(latent_neuts, changes, change_mean, change_std, error_std)
    chain = []
    
    for i in 1:maxiters
        new_error_std = error_std + randn()*prop_bw
        new_change_mean = change_mean + randn()*prop_bw
        new_change_std = change_std + randn()*prop_bw
        new_latent_neuts = propose_some(latent_neuts,prop_bw,0.3)
        new_changes = propose_some(changes,prop_bw,0.3)

        if  new_change_std > 0.05 && new_error_std > 0.05 #Truncating distribution because of stiffness
            new_Lprior = log_prior(new_latent_neuts, new_changes, new_change_mean, new_change_std, new_error_std)
            new_Lpost = new_Lprior + log_lik(founder_neuts, variant_neuts, new_latent_neuts, new_changes, new_error_std)
            if exp(new_Lpost - Lpost) > rand()
                error_std = new_error_std
                change_mean = new_change_mean
                change_std = new_change_std
                latent_neuts = new_latent_neuts
                changes = new_changes
                Lpost = new_Lpost
            end
        end
        if mod(i,2000)==0
            push!(chain,(Lpost,change_mean,change_std,error_std, latent_neuts, changes))
        end
    end
    return chain
end