package jml.optimization;

import static jml.matlab.Matlab.abs;
import static jml.matlab.Matlab.colon;
import static jml.matlab.Matlab.minus;
import static jml.matlab.Matlab.sign;
import static jml.matlab.Matlab.size;
import static jml.matlab.Matlab.sort2;
import static jml.matlab.Matlab.subplus;
import static jml.matlab.Matlab.sumAll;
import static jml.matlab.Matlab.times;
import static jml.matlab.Matlab.zeros;
import static jml.matlab.Matlab.*;

import org.apache.commons.math.linear.BlockRealMatrix;
import org.apache.commons.math.linear.RealMatrix;

/**
 * Compute proj_tC(X) where C = {X: || X ||_1 <= 1}.
 * 
 * @author Mingjie Qian
 * @version 1.0, Oct. 14th, 2013
 */
public class ProjL1 implements Projection {
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		double[][] data = new double[][]{
				{-3.5}, {2.4}, {1.2}, {-0.9}
		};
		RealMatrix X = new BlockRealMatrix(data);
		double t = 1.5;
		display(new ProjL1().compute(t, X));
		
	}

	/**
	 * Compute proj_{tC}(X) where C = {X: || X ||_1 <= 1}.
	 * 
	 * @param t a nonnegative real scalar
	 * 
	 * @param X a real matrix
	 * 
	 * @return proj_{tC}(X) where C = {X: || X ||_1 <= 1}
	 * 
	 */
	public RealMatrix compute(double t, RealMatrix X) {

		if (t < 0) {
			System.err.println("The first input should be a nonnegative real scalar.");
			System.exit(-1);
		}
		
		if (X.getColumnDimension() > 1) {
			System.err.println("The second input should be a vector.");
			System.exit(-1);
		}
		
		RealMatrix res = X.copy();
		
		RealMatrix U = X.copy();
		RealMatrix V = abs(X);
		if (sumAll(V) <= t) {
			res = zeros(size(V));
		}
		int d = size(X)[0];
		sort2(V);
		RealMatrix Delta = V.getSubMatrix(1, d - 1, 0, 0).subtract(
				V.getSubMatrix(0, d - 2, 0, 0));
		RealMatrix S = times(Delta,
				colon(d - 1, -1, 1.0).transpose());
		double a = V.getEntry(d - 1, 0);
		double n = 1;
		double sum = S.getEntry(d - 2, 0);
		for (int j = d - 1; j >= 1; j--) {
			if (sum < t) {
				if (j > 1) {
					sum += S.getEntry(j - 2, 0);
				}
				a += V.getEntry(j - 1, 0);
				n++;
			} else {
				break;
			}
		}
		double alpha = (a - t) / n;
		V = U;
		res = times(sign(V), subplus(minus(abs(V), alpha)));

		return res;
		
	}

}
