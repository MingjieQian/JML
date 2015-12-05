package jml.subspace;

import static jml.matlab.Matlab.disp;
import static jml.matlab.Matlab.eigs;
import static jml.matlab.Matlab.mean;
import static jml.matlab.Matlab.repmat;
import static jml.matlab.Matlab.size;

import java.awt.Color;

import javax.swing.JFrame;

import org.apache.commons.math.linear.BlockRealMatrix;
import org.apache.commons.math.linear.RealMatrix;
import org.math.plot.Plot2DPanel;
import org.math.plot.Plot3DPanel;

/***
 * Principal Component Analysis (PCA).
 * 
 * @author Mingjie Qian
 * @version 1.0, Mar. 29th, 2013
 */
public class PCA extends DimensionalityReduction {

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		double[][] data = {{0, 2, 3, 4}, {2, 0, 4, 5}, {3, 4.1, 5, 6}, {2, 7, 1, 6}};
		RealMatrix X = new BlockRealMatrix(data);
		/*int n = 20;
		int p = 10;
		X = rand(p, n);*/

		int r = 3;
		RealMatrix R = PCA.run(X, r);
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
		plot.addScatterPlot("PCA", Color.RED, x, y);
		plot.addLegend("North");
		
		/*plot.setAxisLabel(0, "this");
		plot.setAxisLabel(1, "that");*/
		// plot.addLabel("this", Color.RED, new double[]{0, 0});

		// put the PlotPanel in a JFrame, as a JPanel
		JFrame frame = new JFrame("A 2D Plot Panel");
		frame.setContentPane(plot);
		frame.setBounds(100, 100, 500, 500);
		frame.setVisible(true);
		
		Plot3DPanel plot3D = new Plot3DPanel();
		plot3D.addScatterPlot("PCA", Color.RED, x, y, z);
		plot3D.addLegend("North");
		
		/*plot.setAxisLabel(0, "this");
		plot.setAxisLabel(1, "that");*/
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
	public PCA(int r) {
		super(r);
	}

	@Override
	public void run() {
		this.R = PCA.run(X, r);
	}
	
	
	/**
	 * PCA.
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
		X = X.subtract(repmat(mean(X, 2), 1, N));
		RealMatrix Psi = X.multiply(X.transpose());
		/*disp(Psi);
		disp(eigs(Psi, r, "lm")[0]);
		disp(eigs(Psi, r, "lm")[1]);*/
		return eigs(Psi, r, "lm")[0].transpose().multiply(X);
		
	}

}
