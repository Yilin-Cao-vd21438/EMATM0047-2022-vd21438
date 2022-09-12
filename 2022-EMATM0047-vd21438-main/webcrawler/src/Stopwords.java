
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.TreeSet;

/**
 * Stop words.  Can be used to filter words.
 * @author James Pope
 */
public class Stopwords
{
    private TreeSet<String> stopwords = null;
    
    /**
     * Creates a new empty set of stop words.
     */
    public Stopwords( )
    {
        this.stopwords = new TreeSet();
    }
    
    /**
     * True if word is in this set of stop words, false otherwise.
     * @param word
     * @return 
     */
    public boolean hasWord( String word )
    {
        return this.stopwords.contains(word);
    }
    
    /**
     * Loads the stop words in the specified file in the stop words set.
     * Duplicates are ignored.
     * @param filename
     * @throws IOException 
     */
    public void load( String filename ) throws IOException
    {
        FileReader fr = new FileReader(filename);
        BufferedReader br = new BufferedReader(fr);
        
        String word = br.readLine();
        while( word != null )
        {
            this.stopwords.add(word);
            word = br.readLine();
        }
        
        br.close();
        fr.close();
    }
    
    /**
     * Loads the stop words in the specified stream in the stop words set.
     * Duplicates are ignored.
     * @param is
     * @throws IOException 
     */
    public void load( InputStream is ) throws IOException
    {
        InputStreamReader fr = new InputStreamReader(is);
        BufferedReader br = new BufferedReader(fr);
        
        String word = br.readLine();
        while( word != null )
        {
            this.stopwords.add(word);
            word = br.readLine();
        }
        
        br.close();
        fr.close();
    }
    
    /**
     * Test main.
     * @param args
     * @throws IOException 
     */
    public static void main(String[] args) throws IOException
    {
        // stop words came from
        //https://www.nltk.org/nltk_data/
        // rainbow stopwords
        // https://gist.github.com/shuson/b3051fae05b312360a18
        
        if( args.length != 1 )
        {
            String u = "  Usage: <text file name with stop words, one per line>";
            String e = "Example: rainbow.txt";
            System.out.println(u);
            System.out.println(e);
            return;
        }
        
        Stopwords sw = new Stopwords();
        sw.load( Stopwords.class.getResourceAsStream("rainbow.txt") );
        
        String word = args[0];
        System.out.printf("hasWord(%s)=%s\n", word, sw.hasWord(word));
    }
}

