
import edu.princeton.std.In;
import edu.princeton.std.StdIn;
import edu.princeton.std.StdOut;

import edu.princeton.algs.ST;
import edu.princeton.algs.Queue;

/*************************************************************************
 *  Compilation:  javac SymbolDigraph.java
 *  Execution:    java SymbolDigraph
 *  Dependencies: ST.java Digraph.java In.java
 *
 *  %  java SymbolDigraph routes.txt " "
 *  JFK
 *     MCO
 *     ATL
 *     ORD
 *  ATL
 *     HOU
 *     MCO
 *  LAX
 *
 *************************************************************************/

public class SymbolDigraph {
    private ST<String, Integer> st;   // string -> index
    private ST<Integer, String> keys; // index  -> string
    private Digraph G;

    public SymbolDigraph() {
        st   = new ST<String, Integer>();
        keys = new ST<Integer, String>();
        G    = new Digraph();
    }

    public void addEdge(String vs, String ws) {
        // Ignore self-loops!!!
        if( vs.equals(ws) ) return;

        if( st.contains(vs) == false ) {
            int id = st.size();
            st.put(vs, id);
            keys.put(id, vs);
        }
        if( st.contains(ws) == false ) {
            int id = st.size();
            st.put(ws, id);
            keys.put(id, ws);
        }

        int vi = st.get(vs);
        int wi = st.get(ws);

        // Ignore parallel edges!!!
        if(!this.G.hasEdge(vi,wi)) {
            this.G.addEdge(vi,wi);
        }
    }

    public Iterable<String> adj(String v) {
        //if (v < 0 || v >= V()) throw new IndexOutOfBoundsException();
        Iterable<Integer> iter = this.G.adj( this.st.get(v) );
        Queue<String> queue = new Queue<String>();
        for( Integer vi : iter )
        {
            queue.enqueue( keys.get(vi) );
        }
        return queue;
    }

    public boolean contains(String s) {
        return st.contains(s);
    }

    public int index(String s) {
        return st.get(s);
    }

    public String name(int v) {
        return keys.get(v);
    }

    public Digraph G() {
        return G;
    }

    public int V() { return G.V(); }
    public int E() { return G.E(); }

    /**
      * Return a string representation of the digraph.
      */
     public String toString() {
         StringBuilder s = new StringBuilder();
         String NEWLINE = System.getProperty("line.separator");
         for ( Integer v : keys.keys() ) {
             String vs = keys.get(v);
             s.append(String.format("%d -> %s: ", v, vs));
             s.append(NEWLINE);
         }
         s.append(NEWLINE);
         return s.toString() + G.toString();
     }


    public static void main(String[] args) {
        String filename  = args[0];
        String delimiter = args[1];
        SymbolDigraph sg = new SymbolDigraph();

        In in = new In(filename);
        while( in.hasNextLine() )
        {
            String line = in.readLine();
            String[] toks = line.split(delimiter);
            String v = toks[0];
            String w = toks[1];
            sg.addEdge(v,w);
        }
        in.close();

        Digraph G = sg.G();
        while (!StdIn.isEmpty()) {
            String t = StdIn.readLine();
            for (int v : G.adj(sg.index(t))) {
                StdOut.println("   " + sg.name(v));
            }
        }
    }
}

