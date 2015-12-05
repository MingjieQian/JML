package jml.optimization;

import org.apache.commons.math.linear.RealMatrix;

public class QPSolution {
	
	public RealMatrix optimizer;
	
	public RealMatrix lambda_opt;
	
	public RealMatrix nu_opt;
	
	public double optimum;
	
	public QPSolution(RealMatrix optimizer, RealMatrix lambda_opt, RealMatrix nu_opt, double optimum) {
		this.optimizer = optimizer;
		this.lambda_opt = lambda_opt;
		this.nu_opt = nu_opt;
		this.optimum = optimum;
	}
	
}
