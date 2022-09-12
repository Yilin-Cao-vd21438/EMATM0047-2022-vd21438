/**
 * Downloads HTML associated with specified ARchive of Our Own URL, 
 * specific to a collection, to a local file.  Collection with a 
 * space in the name should be surrounded with double quotes from
 * the command line.
 * @author James Pope
 */
public class Curl
{
    /**
     * Main entry point into the program.  Reads command line arguments.
     * @param args
     */
    public static void main(String[] args)
    {
        if( args.length != 2 )
        {
            String u = "  Usage: <collection> <page number>";
            String e = "Example: \"Merry Gentry - Laurell K Hamilton\" 2";
            System.out.println(u);
            System.out.println(e);
            return;
        }
        
        //https://archiveofourown.org/tags/Merry%20Gentry%20-%20Laurell%20K%20Hamilton/works?page=2
        String collection = args[0];
        String pageNumber = args[1];
        // Web page prefix
        String urlPrefix = "https://archiveofourown.org/tags/";
        //String archiveUrl = args[0];
        //https://archiveofourown.org/works/22310098
        
        //https://archiveofourown.org/tags/Merry%20Gentry%20-%20Laurell%20K%20Hamilton/works

        collection = collection.replace(" ", "%20");
        String archiveUrl = urlPrefix + collection + "/works?page="+pageNumber;
        
        WebCrawler.readCollection(archiveUrl, collection);
    }

}

