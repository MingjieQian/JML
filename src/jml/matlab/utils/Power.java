package jml.matlab.utils;

import org.apache.commons.math.FunctionEvaluationException;

public class Power implements
		org.apache.commons.math.analysis.UnivariateRealFunction {

	double exponent;
	
	public Power(double exponent) {
		this.exponent = exponent;
	}
	@Override
	public double value(double x) throws FunctionEvaluationException {
		return Math.pow(x, exponent);
	}

}
