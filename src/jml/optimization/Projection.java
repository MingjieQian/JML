package jml.optimization;

import org.apache.commons.math.linear.RealMatrix;

public interface Projection {
	
	public RealMatrix compute(double t, RealMatrix X);

}
