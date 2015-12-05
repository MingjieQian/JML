package jml.optimization;

import static jml.matlab.Matlab.abs;
import static jml.matlab.Matlab.minus;
import static jml.matlab.Matlab.sign;
import static jml.matlab.Matlab.subplus;
import static jml.matlab.Matlab.times;

import org.apache.commons.math.linear.RealMatrix;

/**
 * Compute prox_th(X) where h = || X ||_1.
 * 
 * @author Mingjie Qian
 * @version 1.0, Mar. 11th, 2013
 */
public class ProxL1 implements ProximalMapping {

	/**
	 * Compute prox_th(X) where h = || X ||_1.
	 * 
	 * @param t a nonnegative real scalar
	 * 
	 * @param X a real matrix
	 * 
	 * @return prox_th(X) where h = || X ||_1
	 * 
	 */
	public RealMatrix compute(double t, RealMatrix X) {
		return times(subplus(minus(abs(X), t)), sign(X));
	}

}
