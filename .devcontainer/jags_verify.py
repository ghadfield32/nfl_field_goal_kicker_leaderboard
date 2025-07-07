#!/usr/bin/env python3
"""
Verify that PyJAGS is correctly installed and working.
This script is used by the Docker container health check.
"""

import sys
try:
    import pyjags
    print(f"PyJAGS version: {pyjags.__version__}")
    
    # Create a simple model to verify that PyJAGS works
    code = """
    model {
        # Likelihood
        y ~ dnorm(mu, 1/sigma^2)
        
        # Priors
        mu ~ dnorm(0, 0.001)
        sigma ~ dunif(0, 100)
    }
    """
    
    # Sample data
    data = {'y': 0.5}
    
    # Initialize model with data
    model = pyjags.Model(code, data=data, chains=1, adapt=100)
    print("JAGS model initialized successfully!")
    
    # Sample from the model
    samples = model.sample(200, vars=['mu', 'sigma'])
    print("JAGS sampling completed successfully!")
    
    # Verify the samples
    mu_samples = samples['mu']
    sigma_samples = samples['sigma']
    print(f"mu mean: {mu_samples.mean():.4f}")
    print(f"sigma mean: {sigma_samples.mean():.4f}")
    
    print("PyJAGS verification completed successfully!")
    sys.exit(0)
    
except ImportError:
    print("PyJAGS not found!")
    sys.exit(1)
except Exception as e:
    print(f"Error during PyJAGS verification: {e}")
    sys.exit(1) 
