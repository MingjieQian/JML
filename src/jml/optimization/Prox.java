package jml.optimization;

import org.apache.commons.math.linear.RealMatrix;

/**
 * Compute prox_th(X) where h = 0.
 * 
 * @author Mingjie Qian
 * @version 1.0, Mar. 11th, 2013
 */
public class Prox implements ProximalMapping {

	/**
	 * Compute prox_th(X) where h = 0.
	 * 
	 * @param t a real scalar
	 * 
	 * @param X a real matrix
	 * 
	 * @return prox_th(X) where h = 0
	 * 
	 */
	public RealMatrix compute(double t, RealMatrix X) {
		return X;
	}

}
