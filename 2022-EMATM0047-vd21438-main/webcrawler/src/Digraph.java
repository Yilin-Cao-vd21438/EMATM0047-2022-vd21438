import edu.princeton.algs.Bag;
import edu.princeton.algs.Stack;
import edu.princeton.algs.ST;
import edu.princeton.std.In;
import edu.princeton.std.StdOut;

/*************************************************************************
 *  Compilation:  javac Digraph.java
 *  Execution:    java Digraph filename.txt
 *
 *  A directed graph implemented using a
 *  symbol table of bags.  This allows an arbitrary
 *  number of vertices for a graph where the number
 *  of vertices is unknown (e.g. web crawler)
 *
 *  % java Digraph tinyDG.txt
 *  13 22
 *  0: 5 1
 *  1:
 *  2: 0 3
 *  3: 5 2
 *  4: 3 2
 *  5: 4
 *  6: 9 4 0
 *  7: 6 8
 *  8: 7 9
 *  9: 11 10
 *  10: 12
 *  11: 4 12
 *  12: 9
 *
 *************************************************************************/

/**
 *  The <tt>Digraph</tt> class represents an directed graph
 *  of vertices mapping T to a Bag<T>.
 *  It supports the following operations: add an edge to the graph,
 *  iterate over all of the neighbors incident to a vertex.
 *  Parallel edges and self-loops are permitted.
 */
public class Digraph {
    private int E;
    private ST<Integer, Bag<Integer>> adj;

   /**
     * Create an empty digraph.
     */
    public Digraph() {
        this.adj = new ST<Integer, Bag<Integer>>();
    }


   /**
     * Return the number of vertices in the digraph.
     */
    public int V() {
        return adj.size();
    }

   /**
     * Return the number of edges in the digraph.
     */
    public int E() {
        return E;
    }

   /**
     * Add the directed edge v->w to the digraph.
     *  @throws java.lang.IndexOutOfBoundsException unless both 0 <= v < V and 0 <= w < V
     */
    public void addEdge(int v, int w) {
        //if (v < 0 || v >= V()) throw new IndexOutOfBoundsException();
        //if (w < 0 || w >= V()) throw new IndexOutOfBoundsException();

        // Ignore self-loops!!!
        //if( vs.equals(ws) ) return;

        // Ignore parallel edges!!!
        //if(this.G.hasEdge(v,w)) return;

        // Add v and w if not already in ST
        Bag<Integer> vNeighbors = this.adj.get(v);
        if( vNeighbors == null )
        {
            vNeighbors = new Bag<Integer>();
            this.adj.put(v, vNeighbors);
        }
        vNeighbors.add(w);

        // We also add bag for w if not yet a vertex
        // Though many not have edge leading from it
        Bag<Integer> wNeighbors = this.adj.get(w);
        if( wNeighbors == null )
        {
            wNeighbors = new Bag<Integer>();
            this.adj.put(w, wNeighbors);
        }

        E++;
    }

    public boolean hasEdge( int v, int w ) {
        if( !this.adj.contains(v) ) return false;
        if( !this.adj.contains(w) ) return false;
        for( int n : adj(v) ) {
            if( n == w ) return true;
        }
        return false;
    }

   /**
     * Return the list of vertices pointed to from vertex v as an Iterable.
     * @throws java.lang.IndexOutOfBoundsException unless 0 <= v < V
     */
    public Iterable<Integer> adj(int v) {
        //if (v < 0 || v >= V()) throw new IndexOutOfBoundsException();
        return this.adj.get(v);
    }

   /**
     * Return a string representation of the digraph.
     */
    public String toString() {
        StringBuilder s = new StringBuilder();
        String NEWLINE = System.getProperty("line.separator");
        s.append(this.V() + " " + this.E() + NEWLINE);
        for ( Integer v : this.adj.keys() ) {
            s.append(String.format("%s: ", v));
            for ( Integer w : this.adj.get(v) ) {
                s.append(String.format("%s ", w));
            }
            s.append(NEWLINE);
        }
        return s.toString();
    }

   /**
     * Test client.
     */
    public static void main(String[] args) {
        In in = new In(args[0]);

        Digraph G = new Digraph();

        int V = in.readInt(); // not used, deduced from edges
        int E = in.readInt();
        for (int i = 0; i < E; i++) {
            int v = in.readInt();
            int w = in.readInt();
            G.addEdge(v, w);
        }

        StdOut.println(G);
    }

}

