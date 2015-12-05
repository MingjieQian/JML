package jml.optimization;

import org.apache.commons.math.linear.RealMatrix;

public class PhaseIResult {
	
	public boolean feasible;
	
	public RealMatrix optimizer;
	
	public double optimum;
	
	public PhaseIResult(RealMatrix optimizer, double optimum) {
		this.optimizer = optimizer;
		this.optimum = optimum;
	}
	
	public PhaseIResult(boolean feasible, RealMatrix optimizer, double optimum) {
		this.feasible = feasible;
		this.optimizer = optimizer;
		this.optimum = optimum;
	}
	
}
