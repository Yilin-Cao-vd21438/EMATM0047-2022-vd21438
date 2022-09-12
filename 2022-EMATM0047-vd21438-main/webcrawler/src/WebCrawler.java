/******************************************************************************
 *  Compilation:  javac WebCrawler.java
 *  Execution:    java WebCrawler
 ******************************************************************************/

import edu.princeton.algs.Queue;
import edu.princeton.algs.SET;
//import edu.princeton.std.In;
import edu.princeton.std.StdOut;

import java.io.File;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class WebCrawler
{
    public static void main(String[] args) {

        // timeout connection after 500 miliseconds
        //System.setProperty("sun.net.client.defaultConnectTimeout", "500");
        //System.setProperty("sun.net.client.defaultReadTimeout",    "1000");

        // initial web page
        String archiveUrl = "https://archiveofourown.org";
        //String archiveUrl = args[0];


        //if( true )
        //{
        //    String collectionUrl = "https://archiveofourown.org/tags/Harry%20Potter%20-%20J*d*%20K*d*%20Rowling/works?page=3";
        //    readCollection( collectionUrl );
        //    return;
        //}

        
        // https://archiveofourown.org
        // https://archiveofourown.org/media/Books%20*a*%20Literature/fandoms/tags/1-800-where-r-u%20series%20-%20meg%20cabot/works

        String html = read( archiveUrl + "/media/Books%20*a*%20Literature/fandoms" );

        // NB: Easier to strip all new lines and spaces when using regex.
        html = stripNewlines(html);
        //StdOut.println(html);


       /*************************************************************
        *  Find links of the form: http://xxx.yyy.com
        *  \\w+ for one or more alpha-numeric characters
        *  \\. for dot
        *  could take first two statements out of loop
        *************************************************************/
        // Look for
        // <li>
        //     <a class="tag" href="/tags/The%20African%20Queen%20-%20C*d*%20S*d*%20Forester/works">The African Queen - C. S. Forester</a>(1)</li>
        String regexp ="(<a class=\"tag\" href=)(.)+?(</a>)";
        Pattern pattern = Pattern.compile(regexp);
        Matcher matcher = pattern.matcher(html);

        // find and print all matches
        while (matcher.find())
        {
            try
            {
                String link = matcher.group();
                //StdOut.println(link);

                //------------------------------------------------------------//
                // PARSE OUT URI
                //------------------------------------------------------------//
                String startMark = "href=\"";
                int startIndex = link.indexOf(startMark) + startMark.length();
                String stopMark = "/works";
                int endIndex   = link.indexOf( stopMark ) + stopMark.length();

                String collectionsUri = link.substring( startIndex, endIndex );

                //------------------------------------------------------------//
                // PARSE OUT COUNT AND DERIVE PAGE COUNT
                //------------------------------------------------------------//
                int matchindex = matcher.end();
                System.out.println( "matchindex? " + matchindex );
                int startindex = html.indexOf( "(",  matchindex );
                int stopindex  = html.indexOf( ")", startindex );
                String countString = html.substring(startindex+1, stopindex);
                int count = Integer.parseInt(countString);
                int pages = (count / 20) + 1;
                //String pagestring = html.substring(endregion, endregion+50);
                System.out.println( "count? " + count );
                System.out.println( "pages? " + pages );


                //------------------------------------------------------------//
                // PARSE OUT COLLECTION NAME
                //------------------------------------------------------------//
                StdOut.println("CollectionsUri #####################");

                // Parse out collection name.  Remove spaces and *.
                // Example: /tags/1066%20and%20All%20That%20-%20W*d*%20C*d*%20Sellar%20*a*%20R*d*%20J*d*%20Yeatman/works
                //          1066_and_All_That_-_Wd_Cd_Sellar_a_Rd_Jd_Yeatman
                String collectionName = java.net.URLDecoder.decode( collectionsUri, "UTF-8" );
                collectionName = collectionName.substring(6, collectionName.length() - 6);
                collectionName = collectionName.replace( " ", "_" );
                collectionName = collectionName.replace( "*", "" );

                StdOut.println( collectionName );

                String collectionsUrl = archiveUrl + collectionsUri;
                String escapedCollectionsUrl = removeUTFCharacters( collectionsUrl );
            
                //------------------------------------------------------------//
                // PROCESS EACH COLLECTION
                //------------------------------------------------------------//
                for( int page = 1; page <= pages; page++ )
                {
                    boolean processed = false;
                    while( processed == false )
                    {
                        try
                        {
                            readCollection( escapedCollectionsUrl+"?page="+page, collectionName );
                            processed = true;
                        }
                        catch( Exception e )
                        {
                            e.printStackTrace();
                            // Some time to wait.  Website gets picky when crawling and blocks
                            double wait = StdRandom.uniform( 60000, 120000 );
                            System.out.println( "WAITING: " + (int)wait + " milliseconds" );
                            Thread.sleep( (int)wait );
                        }
                    }
                }

                
            }
            catch( Exception e )
            {
                System.out.println( "EXCEPTION: " + e.getMessage() );
                //System.out.println( "WORKSURL: " + escapedCollectionsUrl );
            }
        }

    }


    /**
     * Visit a "works" page.
     * https://archiveofourown.org/tags/les%20120%20journ%c3%a9es%20de%20sodome
     * %20%7c%20the%20120%20days%20of%20sodom%20-%20marquis%20de%20sade/works
     * 
     * @param escapedCollectionsUrl
     * @param collectionName
     */
    public static boolean readCollection( String escapedCollectionsUrl, String collectionName )
    {
        StdOut.println("ReadCollection: " + escapedCollectionsUrl);

        String collectionsHtml = read( escapedCollectionsUrl );

        // Read will return null if socket connection failed.
        // DEBUGGING
        if( "".equals(collectionsHtml) ) System.out.println("STRING IS EMPTY");
        if( collectionsHtml == null )    System.out.println("STRING IS NULL" );

        if( collectionsHtml == null ) return false;

        // newlines make regex more complicated, remove
        collectionsHtml = stripNewlines(collectionsHtml);

        String regexpBlurb ="(<li class=\"work blurb group\")(.+?)(</dd></dl></li>)";
        Pattern patternBlurb = Pattern.compile(regexpBlurb );


        //StdOut.println("BLURB HTML: " + collectionsHtml);

        //https://archiveofourown.org/tags/the%20aeon&#39;s%20gate%20series%20-%20sam%20sykes/works
        //worksHtml = removeUTFCharacters( worksHtml ).toString();
        int count = 0;
        Matcher matcherBlurb = patternBlurb.matcher(collectionsHtml);
        while( matcherBlurb.find() )
        {
            String blurbHtml = matcherBlurb.group();

            // Check to see if the blurn is English, if so process.
            if( blurbHtml.indexOf( "<dd class=\"language\">English</dd>" ) > -1 )
            {
                handleBlurbHtml( blurbHtml, collectionName );
                //StdOut.println(" ENGLISH  ############################## BEG " + count);
                //StdOut.println(blurbHtml);
                //StdOut.println("          ############################## END " + count);
            }

            // Look for <dd class="language">English</dd>
            //String regexpLang ="(<dd class=\"language\">)(.)+(</dd>)";
            //Pattern patternLang = Pattern.compile(regexpLang);
            // <ol>
            // ...
            // <li class="work blurb group" id="work_20335600" role="article">
            // ...
            // </li>
            // ...
            // </ol>
            // FYI - WILL MISS LAST BLURB
            //String regexpBlurb ="(<li class=\"work blurb group\")(.)+(</ol>|<li class=\"work blurb group\")";
            //String regexpBlurb ="(<li class=\"work blurb group\")(.\\n\\r)?(</li>)";

            count++;
        }
        return true;
    }


    /**
     * Parse the blurp html for further processing.
     * @param blurbHtml
     * @param collectionName
     */
    public static void handleBlurbHtml( String blurbHtml, String collectionName )
    {
        // Look for "Beta'd"  May be Un-Beta'd or Beta'd
        //if( blurbHtml.indexOf( "Beta'd" ) > -1 ||
        //    blurbHtml.indexOf( "beta'd" ) > -1 ||
        //    blurbHtml.indexOf( "Not Beta" ) > -1
        //)
        {
            //StdOut.println(" BETAD  ############################## BEG ");
            //StdOut.println(blurbHtml);

            // <a href="/works/22440748">Clara Potter and the Year it All Went Awry</a>
            String regexp ="(<a href=)(.)+?(</a>)";
            Pattern pattern = Pattern.compile(regexp);
            Matcher matcher = pattern.matcher(blurbHtml);

            // Find and print all matches
            // NB: Default is to show works with chapters on separate html
            // pages. You can change to view whole works by appending view_full_work
            // NB: Some works have adult content and require extra click to "Proceed"
            // but can be avoided by appending ?view_adult=true
            matcher.find();
            String link = matcher.group();
            String uri  = parseLink(link);
            String url = "https://archiveofourown.org" + uri + "/?view_full_work=true&view_adult=true";

            // Get just the number
            String name = uri.replace( "/works/", "works-" );
            
            String worksHtml = read(url);

            // Make sure the collections directory exists
            File collectionsFile = new File(collectionName);
            if( collectionsFile.exists() == false ) collectionsFile.mkdir();

            Out htmlFile = new Out( collectionName + File.separator + name + ".html" );
            htmlFile.print( worksHtml );
            htmlFile.close();

            //StdOut.println( "HTML: " + worksHtml );
            StdOut.println( "TITLE LINK: " + uri );

            //StdOut.println("        ############################## END ");
        }
    }


    /**
     * Takes the URL reads it via the Internet.  You obviously need to have
     * connectivity to the Internet for this to work.
     * @param url
     * @return the html for the url (as though your browser has made the request)
     */
    public static String read( String url )
    {
        In in = new In(url);
        String html = in.readAll();
        return html;
    }

    /**
     * Takes anchor tag <a href="/works/22295164">Halls Decked in Murder</a>
     * and returns the href, i.e. /works/22295164
     * @param link
     * @return the URI in the link
     */
    public static String parseLink( String link )
    {
        String startMark = "href=\"";
        int startIndex = link.indexOf(startMark) + startMark.length();
        String stopMark = "\">";
        int endIndex   = link.indexOf( stopMark );

        String uri = link.substring( startIndex, endIndex );
        return uri;
    }

    /**
     * Convert the HTML Unicode to actual character.  Otherwise, causes
     * issue because not a valid URL.
     * @param data
     * @return data with Unicode literal strings converted to char
     */
    public static String removeUTFCharacters(String data)
    {
        //System.out.println("ESCAPE: " + data);
        //Pattern p = Pattern.compile("\\\\u(\\p{XDigit}{4})"); // \u2122
        Pattern p = Pattern.compile("\\&#(\\p{XDigit}{2});"); //  &#39;
        Matcher m = p.matcher(data);
        StringBuffer buf = new StringBuffer(data.length());
        while (m.find())
        {
            String ch = String.valueOf((char) Integer.parseInt(m.group(1), 10));
            //System.out.println(ch);
            m.appendReplacement(buf, Matcher.quoteReplacement(ch));
        }
        m.appendTail(buf);
        return buf.toString();
    }

    /**
     * Removes all newlines (blackslash n) from the html string.
     * @param html
     * @return html without any newlines
     */
    public static String stripNewlines( String html )
    {
        StringBuilder buf = new StringBuilder();
        String[] tokens = html.split("\n");
        for( String line : tokens )
        {
             buf.append( line.trim() );
        }
        return buf.toString();
    }


    public static void main1(String[] args) {

        // timeout connection after 500 miliseconds
        System.setProperty("sun.net.client.defaultConnectTimeout", "500");
        System.setProperty("sun.net.client.defaultReadTimeout",    "1000");

        // initial web page
        String s = args[0];

        // list of web pages to be examined
        Queue<String> queue = new Queue<String>();
        queue.enqueue(s);

        // set of examined web pages
        SET<String> marked = new SET<String>();
        marked.add(s);

        int visited = 0;

        // breadth first search crawl of web
        while (!queue.isEmpty()) {
            String v = queue.dequeue();
            StdOut.println(v);

            String input = null;
            try {
                In in = new In(v);
                input = in.readAll().toLowerCase();
            }
            catch (IllegalArgumentException e) {
                StdOut.println("[could not open " + v + "]");
                continue;
            }
            catch(NullPointerException e) {
                StdOut.println("[could not open " + v + "]");
                continue;
            }

            // if (input == null) continue;


           /*************************************************************
            *  Find links of the form: http://xxx.yyy.com
            *  \\w+ for one or more alpha-numeric characters
            *  \\. for dot
            *  could take first two statements out of loop
            *************************************************************/
            String regexp = "(http|https)://(\\w+\\.)+(edu|com|gov|org)";
            Pattern pattern = Pattern.compile(regexp);

            Matcher matcher = pattern.matcher(input);

            // find and print all matches
            while (matcher.find()) {
                String w = matcher.group();
                if (!marked.contains(w)) {
                    queue.enqueue(w);
                    marked.add(w);
                }
            }
        }

    }
}

