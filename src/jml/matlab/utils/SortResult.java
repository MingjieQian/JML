package jml.matlab.utils;

import org.apache.commons.math.linear.RealMatrix;

/**
 * <h4>A wrapper for the output of sort function.</h4>
 * There are two data members:<br/>
 * B: sorted values as a {@code RealMatrix} data structure<br/>
 * IX: indices of sorted values in the original matrix<br/>
 */
public class SortResult {
	
	public RealMatrix B;
	public int[][] IX;
	
	public SortResult(RealMatrix B, int[][] IX) {
		this.B = B;
		this.IX = IX;
	}

}
