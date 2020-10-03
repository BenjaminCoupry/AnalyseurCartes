using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace AnalyseurCartes
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();

            /*int x = 800;
            int y = 400;
            Bitmap HEIGHT = new Bitmap(x,y);
            for(int i=0;i<x;i++)
            {
                Console.WriteLine(i);
                for (int j = 0; j < y; j++)
                {
                    double r = (Math.Max(0, perlin(new double[] { i, j }, 42, TypeInterpolation.C2, 7, 0.01, 0.5, 4,1.2)));
                    HEIGHT.SetPixel(i, j, Color.FromArgb((int)(r * 255.0), (int)(r * 255.0), (int)(r * 255.0)));
                }
            }
            HEIGHT.Save("C:/Users/benja/Desktop/cartes/rnd.bmp");
            */
            //Bitmap alt = (Bitmap)Bitmap.FromFile("C:/Users/benja/Desktop/cartes/resize/Alea/rndTerre.bmp");
            //voisinageEau(ref alt).Save("C:/Users/benja/Desktop/cartes/resize/Alea/voisinage.bmp");
            /**
            string Path = "C:/Users/benja/Desktop/cartes/resize/terre/";
            Bitmap rgb = (Bitmap)Image.FromFile(Path + "couleur.jpg");
            int x = rgb.Width;
            int y = rgb.Height;
            Bitmap r = new Bitmap(x, y);
            Bitmap g = new Bitmap(x, y);
            Bitmap b = new Bitmap(x, y);
            for (int i = 0; i < x; i++)
            {
                for (int j = 0; j < y; j++)
                {
                    Color C = rgb.GetPixel(i, j);
                    r.SetPixel(i, j,Color.FromArgb(C.R,C.R,C.R));
                    g.SetPixel(i, j, Color.FromArgb(C.G, C.G, C.G));
                    b.SetPixel(i, j, Color.FromArgb(C.B, C.B, C.B));
                }
            }
            r.Save(Path + "rouge.jpg");
            g.Save(Path + "vert.jpg");
            b.Save(Path + "bleu.jpg");
            **/
            
            string Path = "C:/Users/benja/Desktop/cartes/resize/terre/";
            System.IO.FileInfo[] Files = new System.IO.DirectoryInfo(@Path).GetFiles("*.jpg");
            List<Bitmap> lbp = new List<Bitmap>();
            foreach (System.IO.FileInfo file in Files)
            {
                lbp.Add((Bitmap)Image.FromFile(Path+file.Name));
            }
            int x = lbp.ElementAt(0).Width;
            int y= lbp.ElementAt(0).Height;
            using (System.IO.StreamWriter file =
            new System.IO.StreamWriter(Path+"valeurs.csv"))
            {
                string str = "";
                for (int k = 0; k < lbp.Count; k++)
                {
                    str = str + System.IO.Path.GetFileNameWithoutExtension(Files.ElementAt(k).Name)+";";
                }
                file.WriteLine(str);
                for (int i = 0; i < x; i++)
                {
                    for (int j = 0; j < y; j++)
                    {
                         str = "";
                        for(int k=0;k<lbp.Count;k++)
                        {
                            str = str+lbp.ElementAt(k).GetPixel(i,j).R+";";
                        }
                        file.WriteLine(str);
                    }
                }
            }
            
        }
        public static Bitmap voisinageEau(ref Bitmap heights)
        {
            int x = heights.Width;
            int y = heights.Height;
            int R = 100;
            float[,] retour_ = new float[x, y];
            Bitmap retour = new Bitmap(x, y);
            float max = 0;
            int n = 0;
            for (int i = 0; i < x; i++)
            {
                for (int j = 0; j < y; j++)
                {
                    float P = 0;
                    if(n%60==0)
                    Console.WriteLine(n + "/" + x * y);
                    for (int k = -R/2; k < R/2; k+=1)
                    {
                        for (int l = -R/2; l < R/2; l+=1)
                        {
                            int ir = ((k + i) % x + x) % x;
                            int jr = ((l + j) % y + y) % y;
                            if (heights.GetPixel(ir, jr).Equals(Color.FromArgb(255, 0, 0, 0)))
                            {
                                float r = (float)Math.Sqrt(k * k + l * l);
                                if (r <= R)
                                {
                                    P += (float)(1.0 / (1.0 + r));
                                }
                            }
                        }
                    }
                    if (P > max)
                    {
                        max = P;
                    }
                    retour_[i, j] = P;
                    n++;
                }
            }

            for (int i = 0; i < x; i++)
            {
                for (int j = 0; j < y; j++)
                {
                    int teinte = (int)(255.0 * retour_[i, j] / max);
                    retour.SetPixel(i, j, Color.FromArgb(teinte, teinte, teinte));
                }
            }
            return retour;
        }
        public static Bitmap Distances_Noir(ref Bitmap heights)
        {
            int x = heights.Width;
            int y = heights.Height;
            float[,] retour_ = new float[x, y];
            Bitmap retour = new Bitmap(x, y);
            float max = 0;
            int n = 0;
            for (int i = 0; i < x; i++)
            {
                for (int j = 0; j < y; j++)
                {
                    n++;
                    Console.WriteLine(n + "/" + x*y);
                    if (heights.GetPixel(i, j).Equals(Color.FromArgb(255, 0, 0, 0)))
                    {
                        retour_[i, j] = 0;
                    }
                    else
                    {
                        List<float> dists = new List<float>();
                        int r = 1;
                        while (dists.Count == 0)
                        {
                            for (int i_ = -r; i_ < r; i_++)
                            {
                                for (int j_ = -r; j_ < r; j_++)
                                {
                                    int ir = ((i_ + i) % x+x)%x;
                                    int jr = ((j_ + j) % y+y)%y;
                                    if (heights.GetPixel(ir, jr).Equals(Color.FromArgb(255, 0, 0, 0)))
                                    {
                                        float dist = (float)Math.Sqrt(Math.Pow(i_, 2) + Math.Pow(j_, 2));
                                        if (dist <= r)
                                        {
                                            if (!dists.Contains(dist))
                                            {
                                                dists.Add(dist);
                                            }
                                        }
                                    }
                                }
                            }
                            r++;
                        }
                        float dm = dists.Min();
                        if (dm > max)
                        {
                            max = dm;
                        }
                        retour_[i, j] = dm;
                    }
                }
            }

            for (int i = 0; i < x; i++)
            {
                for (int j = 0; j < y; j++)
                {
                    int teinte = (int)(255.0*retour_[i,j]/max);
                    retour.SetPixel(i, j, Color.FromArgb(teinte, teinte, teinte));
                }
            }
            return retour;
        }
        public static int dist_droite_noir(ref Bitmap heights, int x,int y, int sgn)
        {
            int k = 0;
            int X = heights.Width;
            int xr = x;
            while(!heights.GetPixel(xr, y).Equals(Color.FromArgb(255, 0, 0, 0)))
            {
                xr = ((x+k) % X + X) % X;
                k += sgn;
                if(Math.Abs(k)>X)
                {
                    return X;
                }
            }
            return xr;
        }
        public static Bitmap ParcoursEau( Bitmap heights)
        {
            int x = heights.Width;
            int y = heights.Height;
            float[,] retour_ = new float[x, y];
            Bitmap retour = new Bitmap(x, y);
            float max = 0;
            for (int i = 0; i < x; i++)
            {
                Console.WriteLine(i + "/" + x);
                for (int j = 0; j < y; j++)
                {
                    Application altitude = pos => heights.GetPixel((int)pos[0], j).R;
                    int sgn;
                    if (j > y /2)
                    {
                        sgn = 1;
                    }
                    else
                    {
                        sgn = -1;
                    }
                    int Xintegr=dist_droite_noir(ref heights,i,j,sgn);
                    int x0 = Math.Min(Xintegr, i);
                    int dist = Math.Abs(Xintegr-i);
                    float val = Math.Abs((float)integrer(new double[2] { x0,j }, dist, 0, 0.5, altitude));
                    if (val>max)
                    {
                        max = val;
                    }
                    retour_[i, j] = val;
                }
            }

            for (int i = 0; i < x; i++)
            {
                for (int j = 0; j < y; j++)
                {
                    int teinte = (int)(255.0 * retour_[i, j] / max);
                    retour.SetPixel(i, j, Color.FromArgb(teinte, teinte, teinte));
                }
            }
            return retour;
        }
        public static double integrer(double[] x0, double distIntegration, int dimIntegration, double delta, Application f)
        {
            //Integrer la fonction f selon la dim dimIntegration sur une longueur distIntegration en partant de x0, avec un pas de delta
            double somme = 0.0;
            int NbPas = (int)(distIntegration / delta);
            double[] x = copiedb(x0);
            for (int i = 0; i < NbPas; i++)
            {
                somme += delta * f(x);
                x[dimIntegration] += delta;
            }
            return somme;

        }
        private static double[] copiedb(double[] V1)
        {
            //renvoie une copie de V1
            int Taille = V1.Length;
            double[] Resultat = new double[Taille];
            for (int i = 0; i < Taille; i++)
            {
                Resultat[i] = V1[i];
            }
            return Resultat;
        }
        public delegate double Application(double[] x);








        public enum TypeInterpolation { Lineaire, Cosinus, Hermite, C2 };
        //Fonctions pseudoAlea
        private static double pseudo_Alea(int N, int seed)
        {
            //Retourne un double pseudo aleatoire N->R
            N = N + seed * 58900;
            N = (N << 13) ^ N;
            N = (N * (N * N * 15731 + 789221)) + 1376312589;
            return 1.0 - (N & 0x7fffffff) / 1073741824.0;
        }
        private static double pseudo_Alea_Rn(int[] V, int seed)
        {
            //Retourne un double pseudo aleatoire N^k->R
            int Taille = V.Length;
            double tmp = 0.0;
            for (int i = 0; i < Taille; i++)
            {
                tmp = (tmp * 850000.0);
                tmp = pseudo_Alea((int)tmp + V[i], seed);
            }
            return tmp;
        }
        private static double[] vect_pseudo_Alea_Rn(int[] V, int seed)
        {
            //Renvoie un vecteur de l'hypershere de dimention n, n etant la dimention de v
            int Taille = V.Length;
            double[] Retour = new double[Taille];
            double Norme;
            do
            {
                //Generer un vecteur dans l'hypercube
                for (int i = 0; i < Taille; i++)
                {
                    Retour[i] = pseudo_Alea_Rn(V, seed);
                    seed += 1;
                }
                Norme = normeVect(Retour);
            }
            while (Norme == 0 || Norme > 1.0);
            //Le vecteur est dans l'hyperboule, le normaliser
            multiplierVect(ref Retour, 1.0 / Norme);
            return Retour;
        }
        private static double[] vect_pseudo_Alea(int N, int Taille, int seed)
        {
            //Renvoie un vecteur de l'hypershere de dimention Taille
            double[] Retour = new double[Taille];
            double Norme;
            do
            {
                //Generer un vecteur dans l'hypercube
                for (int i = 0; i < Taille; i++)
                {
                    Retour[i] = pseudo_Alea(N, seed);
                    seed += 1;
                }
                Norme = normeVect(Retour);
            }
            while (Norme == 0 || Norme > 1.0);
            //Le vecteur est dans l'hyperboule, le normaliser
            multiplierVect(ref Retour, 1.0 / Norme);
            return Retour;
        }

        //Interpolation
        private static double fonctionsInterp(double x, TypeInterpolation Interp)
        {
            //Renvoie le coefficient d'interpolation k [0,1] en fonction du parcours x
            double result;
            switch (Interp)
            {
                case TypeInterpolation.Lineaire:
                    result = x;
                    break;
                case TypeInterpolation.Cosinus:
                    result = 0.5 * (1.0 - Math.Cos(x * Math.PI));
                    break;
                case TypeInterpolation.Hermite:
                    result = 3.0 * x * x - 2.0 * x * x * x;
                    break;
                case TypeInterpolation.C2:
                    result = 6.0 * x * x * x * x * x - 15.0 * x * x * x * x + 10.0 * x * x * x;
                    break;
                default:
                    result = 0.0;
                    break;
            }
            return result;
        }
        private static double interpolerEspace(double[] Position, int[] PosGrille, int N0, int seed, TypeInterpolation TI)
        {
            //Permet d'obtenir une interpolation pour une sous dim, a partir d'une selection des coordonees du point de ref pour les dim superieures
            if (N0 < 0)
            {
                //La valeur voulue pour un point de la grille est le produit scalaire entre le gradient a la position de la grille voulue, et la distance a cette position
                double[] dist = recentrer(PosGrille, Position);
                return produitScalaire(dist, vect_pseudo_Alea_Rn(PosGrille, seed));
            }
            else
            {
                //On choisit la dim N0 (2 choix) et on interpole les deux options entre elles
                double x = Position[N0];
                //projection sur la dim N0 et recuperation du coeff d'interpolation
                double k = fonctionsInterp(x - Math.Floor(x), TI);
                int[] P1 = copie(PosGrille);
                int[] P2 = copie(PosGrille);
                //fixer la valeur pour la dim N0
                P1[N0] = (int)Math.Floor(Position[N0]);
                P2[N0] = (int)Math.Floor(Position[N0]) + 1;
                //interpolation
                double b1 = interpolerEspace(Position, P1, N0 - 1, seed, TI);
                double b2 = interpolerEspace(Position, P2, N0 - 1, seed, TI);
                double inter = ((1.0 - k) * b1 + k * b2);
                return inter;
            }
        }
        private static double perlinSimple(ref double[] Position, int seed, TypeInterpolation TI)
        {
            //Obtient un bruit de perlin itere une fois et de frequence 1
            return Math.Min(Math.Max(interpolerEspace(Position, new int[Position.Length], Position.Length - 1, seed, TI), -1.0), 1.0);
        }
        public static double perlin(double[] Position, int seed, TypeInterpolation TI, int nbOctaves, double f0, double Attenuation, double Decalage, double puissance, int nbPlateaux, double k_plateaux)
        {
            //Renvoie un bruit de perlin a n dim pour l'interpolation specifiee, lissage par la puisssance "puissance"
            //nbPlateaux, nombre de plages de valeurs prises possibles. mettre à 0 pour toutes les valeurs possibles
            //k_plateaux est la pente des plateaux [0,1]
            double Resultat = 0.0;
            double Amplitude = 1.0;
            double f = f0;
            double[] shift = new double[Position.Length];
            double[] Pos = new double[Position.Length];
            double sommeAmp = 0;
            for (int i = 0; i < nbOctaves; i++)
            {
                shift = vect_pseudo_Alea(i * 452237 + 700849, Position.Length, seed);
                multiplierVect(ref shift, Decalage * pseudo_Alea(i * 89746 + 6577, seed));
                Pos = copie(Position);
                shifter(ref Pos, ref shift);
                multiplierVect(ref Pos, f);
                Resultat += perlinSimple(ref Pos, seed, TI) * Amplitude;
                sommeAmp += Amplitude;
                Amplitude *= Attenuation;
                f *= 2;
            }
            Resultat = Resultat / sommeAmp;
            Resultat = Math.Sign(Resultat) * Math.Pow(Math.Abs(Resultat), puissance);
            if (nbPlateaux > 0)
            {
                double s = 0;
                if (k_plateaux != 0)
                {
                    s = k_plateaux * (1.0 / (float)nbPlateaux) * ((float)nbPlateaux * Resultat - Math.Floor((float)nbPlateaux * Resultat));
                }
                Resultat = (1.0 / (float)nbPlateaux) * Math.Floor((float)nbPlateaux * Resultat);
                Resultat += s;
            }
            return Resultat;
        }
        public static double perlin(double[] Position, int seed, TypeInterpolation TI, int nbOctaves, double f0, double Attenuation, double Decalage, double puissance, int nbPlateaux)
        {
            //Renvoie un bruit de perlin a n dim pour l'interpolation specifiee, lissage par la puisssance "puissance"
            //nbPlateaux, nombre de valeurs prises possibles
            return perlin(Position, seed, TI, nbOctaves, f0, Attenuation, Decalage, puissance, nbPlateaux, 0.0);
        }
        public static double perlin(double[] Position, int seed, TypeInterpolation TI, int nbOctaves, double f0, double Attenuation, double Decalage, double puissance)
        {
            //Renvoie un bruit de perlin a n dim pour l'interpolation specifiee, lissage par la puisssance "puissance"
            return perlin(Position, seed, TI, nbOctaves, f0, Attenuation, Decalage, puissance, 0, 0);
        }
        public static double perlin(double[] Position, int seed, TypeInterpolation TI, int nbOctaves, double f0, double Attenuation, double Decalage)
        {
            //Renvoie un bruit de perlin a n dim pour l'interpolation specifiee
            return perlin(Position, seed, TI, nbOctaves, f0, Attenuation, Decalage, 1.0, 0, 0);
        }
        public static double perlin(double[] Position, int seed, int nbOctaves, double f0, double Attenuation, double Decalage)
        {
            //Renvoie un bruit de perlin a n dim pour l'interpolation C2
            return perlin(Position, seed, TypeInterpolation.C2, nbOctaves, f0, Attenuation, Decalage, 1.0, 0, 0);
        }

        //Fonctions vectorielles
        private static double produitScalaire(double[] V1, double[] V2)
        {
            //Produit scalaire de deux vecteurs
            int Taille = Math.Min(V1.Length, V2.Length);
            double Somme = 0.0;
            for (int i = 0; i < Taille; i++)
            {
                Somme += V1[i] * V2[i];
            }
            return Somme;
        }
        private static double normeVect(double[] V)
        {
            //Norme du vecteur v
            return Math.Sqrt(produitScalaire(V, V));
        }
        private static void multiplierVect(ref double[] V, double k)
        {
            //Multiplier le vecteur par k
            int Taille = V.Length;
            for (int i = 0; i < Taille; i++)
            {
                V[i] *= k;
            }
        }
        private static void shifter(ref double[] V1, ref double[] V2)
        {
            //renvoie le vecteur V1 = V1 +V2
            int Taille = Math.Min(V1.Length, V2.Length);
            for (int i = 0; i < Taille; i++)
            {
                V1[i] += V2[i];
            }
        }
        private static double[] recentrer(int[] V1, double[] V2)
        {
            //renvoie le vecteur V2-V1
            int Taille = Math.Min(V1.Length, V2.Length);
            double[] Resultat = new double[Taille];
            for (int i = 0; i < Taille; i++)
            {
                Resultat[i] = V2[i] - (double)V1[i];
            }
            return Resultat;
        }
        private static int[] copie(int[] V1)
        {
            //renvoie une copie de V1
            int Taille = V1.Length;
            int[] Resultat = new int[Taille];
            for (int i = 0; i < Taille; i++)
            {
                Resultat[i] = V1[i];
            }
            return Resultat;
        }
        private static double[] copie(double[] V1)
        {
            //renvoie une copie de V1
            int Taille = V1.Length;
            double[] Resultat = new double[Taille];
            for (int i = 0; i < Taille; i++)
            {
                Resultat[i] = V1[i];
            }
            return Resultat;
        }
    }
}
