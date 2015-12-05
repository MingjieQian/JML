package jml.optimization;

import org.apache.commons.math.linear.RealMatrix;

public interface ProximalMapping {
	
	public RealMatrix compute(double t, RealMatrix X);

}
