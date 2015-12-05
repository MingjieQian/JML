package jml.optimization;

import static jml.matlab.Matlab.subplus;

import org.apache.commons.math.linear.RealMatrix;

/**
 * Compute prox_th(X) where h = I_+(X).
 * 
 * @author Mingjie Qian
 * @version 1.0, Mar. 11th, 2013
 */
public class ProxPlus implements ProximalMapping {

	/**
	 * Compute prox_th(X) where h = I_+(X).
	 * 
	 * @param t a real scalar
	 * 
	 * @param X a real matrix
	 * 
	 * @return prox_th(X) where h = I_+(X)
	 * 
	 */
	public RealMatrix compute(double t, RealMatrix X) {
		return subplus(X);
	}

}
