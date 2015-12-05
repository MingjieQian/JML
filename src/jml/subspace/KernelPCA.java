package jml.subspace;

import static jml.matlab.Matlab.diag;
import static jml.matlab.Matlab.disp;
import static jml.matlab.Matlab.eigs;
import static jml.matlab.Matlab.eye;
import static jml.matlab.Matlab.mtimes;
import static jml.matlab.Matlab.ones;
import static jml.matlab.Matlab.power;
import static jml.matlab.Matlab.rdivide;
import static jml.matlab.Matlab.size;

import java.awt.Color;

import javax.swing.JFrame;

import jml.kernel.Kernel;

import org.apache.commons.math.linear.BlockRealMatrix;
import org.apache.commons.math.linear.RealMatrix;
import org.math.plot.Plot2DPanel;
import org.math.plot.Plot3DPanel;

/***
 * Kernel PCA
 * 
 * @author Mingjie Qian
 * @version 1.0, Mar. 29th, 2013
 */
public class KernelPCA extends DimensionalityReduction {

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		double[][] data = {
				{0, 2, 3, 4}, 
				{2, 0, 4, 5}, 
				{3, 4.1, 5, 6}, 
				{2, 7, 1, 6}
				};
		RealMatrix X = new BlockRealMatrix(data);
		/*int n = 20;
		int p = 10;
		X = rand(p, n);*/

		int r = 3;
		RealMatrix R = KernelPCA.run(X, r);
		disp("Original Data:");
		disp(X);
		disp("Reduced Data:");
		disp(R);
		
		double[] x = R.getRow(0);
		double[] y = R.getRow(1);
		double[] z = R.getRow(2);
		
		// create your PlotPanel (you can use it as a JPanel)
		Plot2DPanel plot = new Plot2DPanel();

		// add a line plot to the PlotPanel
		// plot.addLinePlot("my plot", Color.RED, x, y);
		
		// add a scatter plot to the PlotPanel
		plot.addScatterPlot("KernelPCA", Color.RED, x, y);
		plot.addLegend("North");
		
		plot.setAxisLabel(0, "this");
		plot.setAxisLabel(1, "that");
		// plot.addLabel("this", Color.RED, new double[]{0, 0});

		// put the PlotPanel in a JFrame, as a JPanel
		JFrame frame = new JFrame("A 2D Plot Panel");
		frame.setContentPane(plot);
		frame.setBounds(100, 100, 500, 500);
		frame.setVisible(true);
		
		Plot3DPanel plot3D = new Plot3DPanel();
		plot3D.addScatterPlot("KernelPCA", Color.RED, x, y, z);
		plot3D.addLegend("North");
		
		plot.setAxisLabel(0, "this");
		plot.setAxisLabel(1, "that");
		// plot.addLabel("this", Color.RED, new double[]{0, 0});

		// put the PlotPanel in a JFrame, as a JPanel
		JFrame frame3D = new JFrame("A 3D Plot Panel");
		frame3D.setContentPane(plot3D);
		frame3D.setBounds(100, 100, 500, 500);
		frame3D.setVisible(true);
		
	}
	
	/**
	 * Constructor.
	 * 
	 * @param r number of dimensions to be reduced to
	 * 
	 */
	public KernelPCA(int r) {
		super(r);

	}

	@Override
	public void run() {
		this.R = run(X, r);
	}
	
	/**
	 * Kernel PCA.
	 * 
	 * @param X a d x n data matrix
	 * 
	 * @param r number of dimensions to be reduced to
	 * 
	 * @return a r x n matrix which is the r dimensional 
	 *         representation of the given n examples
	 *         
	 */
	public static RealMatrix run(RealMatrix X, int r) {
		
		int N = size(X, 2);
		
		RealMatrix H = eye(N).subtract(rdivide(ones(N, N), N));
		double sigma = 1.0;
		RealMatrix K = Kernel.calcKernel("rbf", sigma, X);
		RealMatrix Psi = H.multiply(K).multiply(H);
		// disp(Psi);
		RealMatrix[] UD = eigs(Psi, r, "lm");
		RealMatrix U = UD[0];
		RealMatrix D = UD[1];
		/*disp(U);
		disp(U.transpose().multiply(U));
		disp(D);
		disp(Psi.multiply(U));
		disp(U.multiply(D));
		disp("UDU':");
		disp(U.multiply(D).multiply(U.transpose()));*/
		D = diag(rdivide(1, power(diag(D), 0.5)));
		U = mtimes(U, D);
		
		return U.transpose().multiply(K);
		
	}

}
