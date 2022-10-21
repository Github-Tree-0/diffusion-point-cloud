#include <cstdio>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <cassert>
#include <cmath>
#include <vector>
#include <set>
#include <utility>
#include <queue>

#define For(i, a, b) for(register int i = a, ___u = b; i <= ___u; ++i)
#define ForDown(i, a, b) for(register int i = b, ___d = a; i >= ___d; --i)
#define cmax(i, j) ((i) < (j) ? (i) = (j) : (i))
#define cmin(i, j) ((i) > (j) ? (i) = (j) : (i))
​
namespace pb_ds
{
	using io::P;
 
	const int MAXN = 2000010;
	struct Po {double d[3];} p[MAXN];
	int D;
	inline bool cmp(const Po& a, const Po& b)
	{
		return a.d[D] < b.d[D];
	}
	struct Node {double min[3], max[3]; int size;} T[MAXN << 2];
	inline void build(int at, int l, int r, int d)
	{
		int m = (l + r) >> 1;
		D = d;
		std::nth_element(p + l, p + m, p + r + 1, cmp);
		T[at].size = r - l + 1;
		For(i, 0, 3)
		{
			T[at].min[i] = 1e100;
			T[at].max[i] = -1e100;
		}
		For(i, l, r) For(j, 0, 2)
		{
			cmin(T[at].min[j], p[i].d[j]);
			cmax(T[at].max[j], p[i].d[j]);
		}
		d-- ? 1 : d += 3;
		if(l < r)
		{
			build(at << 1, l, m, d);
			build(at << 1 | 1, m + 1, r, d);
		}
	}
	Po cur;
	int ans;
	inline void query(int at)
	{
		bool not_cover = 0;
		For(i, 0, 3)
			if(T[at].max[i] > cur.d[i])
			{
				not_cover = 1;
				if(T[at].min[i] > cur.d[i])
					return;
			}  
		 
		if(not_cover)
		{
			if(T[at << 1].size) query(at << 1);
			if(T[at << 1 | 1].size) query(at << 1 | 1);
		}
		else ans += T[at].size;
	}
 
	inline void main()
	{
		int n = F();
		For(i, 1, n)
			p[i] = (Po) {G(), G(), G()};
		build(1, 1, n, 0); 
		
	}  
}