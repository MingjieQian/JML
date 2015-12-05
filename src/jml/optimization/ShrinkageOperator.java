package jml.optimization;

import static jml.matlab.Matlab.abs;
import static jml.matlab.Matlab.minus;
import static jml.matlab.Matlab.sign;
import static jml.matlab.Matlab.subplus;
import static jml.matlab.Matlab.times;

import org.apache.commons.math.linear.RealMatrix;

/***
 * Soft-thresholding (shrinkage) operator, which is defined as
 * S_{t}[x] = argmin_u 1/2 * || u - x ||^2 + t||x||_1</br>
 * which is actually prox_{t||.||_1}(x). The analytical form is</br>
 * S_{t}[x] =</br>
 * | x - t, if x > t</br>
 * | x + t, if x < -t</br>
 * | 0, otherwise</br>
 * 
 * @author Mingjie Qian
 * @version 1.0, Nov. 19th, 2013
 */
public class ShrinkageOperator {
	
	/**
	 * Soft-thresholding (shrinkage) operator, which is defined as
	 * S_{t}[x] = argmin_u 1/2 * || u - x ||^2 + t||x||_1</br>
	 * which is actually prox_{t||.||_1}(x). The analytical form is</br>
	 * S_{t}[x] =</br>
	 * | x - t, if x > t</br>
	 * | x + t, if x < -t</br>
	 * | 0, otherwise</br>
	 * 
	 * @param X a real matrix
	 * 
	 * @param t threshold
	 * 
	 * @return argmin_u 1/2 * || u - x ||^2 + t||x||_1
	 */
	public static RealMatrix shrinkage(RealMatrix X, double t) {
		return times(subplus(minus(abs(X), t)), sign(X));
	}

}
