import java.util.Scanner;
import java.util.Stack;
import java.util.Vector;

class BoardState
{
	int mat[][];
	int goalstate[][];
	private int r,c;
	public BoardState()
	{
		Scanner scan=new Scanner(System.in);
		System.out.println("Enter start config: ");
		mat=new int[3][3];
		goalstate=new int[3][3];
		for(int i=0;i<3;i++)
		{
			for(int j=0;j<3;j++)
			{
				mat[i][j]= scan.nextInt();
				if(mat[i][j]==0)
				{
					r=i; c=j;
				}
			}
		}
		System.out.println("Enter goal config: ");
		for(int i=0;i<3;i++)
		{
			for(int j=0;j<3;j++)
			{
				goalstate[i][j]=scan.nextInt();
			}
		}
		scan.close();
	}
	public BoardState(BoardState B)
	{
		mat=new int[3][3];
		goalstate=new int[3][3];
		for(int i=0;i<3;i++)
		{
			for(int j=0;j<3;j++)
			{
				mat[i][j]= B.mat[i][j];
				if(mat[i][j]==0)
				{
					r=i; c=j;
				}
			}
		}
		for(int i=0;i<3;i++)
		{
			for(int j=0;j<3;j++)
			{
				goalstate[i][j]= B.goalstate[i][j];
			}
		}
	}
	public void show()
	{
		System.out.println("\nBoard:");
		for(int i=0;i<3;i++)
		{
			for(int j=0;j<3;j++)
			{
				System.out.print(mat[i][j]);
			}
			System.out.println();
		}
	}
	public int heuristic()
	{
		int heuristic=0;
		for(int i=0;i<3;i++)
		{
			for(int j=0;j<3;j++)
			{
				if(	mat[i][j]!=goalstate[i][j] )
				{
					heuristic++;
				}

			}
		}
		return heuristic;
	}

	//Movements
	public BoardState moveup()
	{
		if(r==0)
			return new BoardState(this);
		BoardState B=new BoardState(this);
		int temp;
		temp=B.mat[r-1][c];
		B.mat[r-1][c]=B.mat[r][c];
		B.mat[r][c]=temp;
		B.r=B.r-1;
		return B;
	}
	public BoardState movedown()
	{
		if(r==2)
			return new BoardState(this);
		BoardState B=new BoardState(this);
		int temp;
		temp=B.mat[r+1][c];
		B.mat[r+1][c]=B.mat[r][c];
		B.mat[r][c]=temp;
		B.r=B.r+1;
		return B;
	}
	public BoardState moveleft()
	{
		if(c==0)
			return new BoardState(this);
		BoardState B=new BoardState(this);
		int temp;
		temp=B.mat[r][c-1];
		B.mat[r][c-1]=B.mat[r][c];
		B.mat[r][c]=temp;
		B.c=B.c-1;
		return B;
	}
	public BoardState moveright()
	{
		if(c==2)
			return new BoardState(this);
		BoardState B=new BoardState(this);
		int temp;
		temp=B.mat[r][c+1];
		B.mat[r][c+1]=B.mat[r][c];
		B.mat[r][c]=temp;
		B.c=B.c+1;
		return B;
	}

	public boolean isSolved()
	{
		if(heuristic() == 0)
			return true;
		return false;
	}
}


class Ayster
{
	Stack <BoardState> CL;
	public Ayster()
	{
		CL=new Stack <BoardState>();
	}
	
	public void run(BoardState B)
	{
		BoardState next, curr;
		curr=B;
		boolean solved=false,found=false;
		int hval;
		do
		{
			if(curr.isSolved())
			{
				solved=true;
				break;
			}
			found=false;
			hval=curr.heuristic();
			
			//Check Move up
			System.out.println("Current Heuristic: "+hval);
			curr.show();
			System.out.println("UP Heuristic: "+curr.moveup().heuristic());
			curr.moveup().show();
			System.out.println("DOWN Heuristic: "+curr.movedown().heuristic());
			curr.movedown().show();
			System.out.println("LEFT Heuristic: "+curr.moveleft().heuristic());
			curr.moveleft().show();
			System.out.println("RIGHT Heuristic: "+curr.moveright().heuristic());
			curr.moveright().show();
			if(curr.moveup().heuristic()<hval)
			{
				next=curr.moveup();
				hval=next.heuristic();
				CL.push(curr);
				curr=next;
				found=true;
				System.out.println("\n\t\t\t\tMoved Up");
			}
			//Check Move down
			else if(curr.movedown().heuristic()<hval)
			{
				next=curr.movedown();
				hval=next.heuristic();
				CL.push(curr);
				curr=next;
				found=true;
				System.out.println("\n\t\t\t\tMoved Down");
			}
			//Check Move left
			else if(curr.moveleft().heuristic()<hval)
			{
				next=curr.moveleft();
				hval=next.heuristic();
				CL.push(curr);
				curr=next;
				found=true;
				System.out.println("\n\t\t\t\tMoved Left");
			}
			
			//Check Move right
			else if(curr.moveright().heuristic()<hval)
			{
				next=curr.moveright();
				hval=next.heuristic();
				CL.push(curr);
				curr=next;
				found=true;
				System.out.println("\n\t\t\t\tMoved RIght");
			}
			
		} while( found );
		
		if(solved)
		{
			System.out.println("Avenger theme starts");
			curr.show();
		}
		else
		{
			System.out.println("Hag Diya");
			curr.show();
		}
	}
}


public class AysterHillClimbing 
{
	public static void main(String args[])
	{
		BoardState B=new BoardState();
		/*B.show();
		B.heuristic();
		B.moveup();
		B.show();
		B.heuristic();
		B.movedown();
		B.show();
		B.heuristic();
		B.moveleft();
		B.show();
		B.heuristic();
		B.moveright();
		B.show();
		B.heuristic();*/
		Ayster Algorithm=new Ayster();
		Algorithm.run(B);
	}
}

