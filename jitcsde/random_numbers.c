/* Random-number generator adapted from NumPy’s randomkit.c and randomkit.h

	Copyright (c) 2003-2005, Jean-Sebastien Roy (js@jeannot.org); 2017, Gerrit Ansmann
	
	The rk_random and rk_seed functions algorithms and the original design of the Mersenne Twister RNG:
	
		Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
		All rights reserved.
		
		Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
		
		1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
		
		2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
		
		3. The names of its contributors may not be used to endorse or promote products derived from this software without specific prior written permission.
		
		THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
	
	Original algorithm for the implementation of rk_interval function from Richard J. Wagner's implementation of the Mersenne Twister RNG, optimised by Magnus Jonsson.
	
	Constants used in the rk_double implementation by Isaku Wada.
	
	Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
	
	The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
	
	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#define RK_STATE_LEN 624

typedef struct rk_state_
{
	unsigned long key[RK_STATE_LEN];
	int pos;
}
rk_state;

// Initialize the RNG state using the given seed.
void rk_seed(unsigned long seed, rk_state * const state)
{
	seed &= 0xffffffffUL;
	
	/* Knuth's PRNG as used in the Mersenne Twister reference implementation */
	for (int pos = 0; pos < RK_STATE_LEN; pos++)
	{
		state->key[pos] = seed;
		seed = (1812433253UL * (seed ^ (seed >> 30)) + pos + 1) & 0xffffffffUL;
	}
	state->pos = RK_STATE_LEN;
}

/* Magic Mersenne Twister constants */
#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL
#define UPPER_MASK 0x80000000UL
#define LOWER_MASK 0x7fffffffUL

/*
	Slightly optimised reference implementation of the Mersenne Twister
	Note that regardless of the precision of long, only 32 bit random integers are produced
*/
unsigned long rk_random(rk_state * const state)
{
	unsigned long y;
	
	if (state->pos == RK_STATE_LEN)
	{
		int i;
		
		for (i = 0; i < N-M; i++)
		{
			y = (state->key[i] & UPPER_MASK) | (state->key[i+1] & LOWER_MASK);
			state->key[i] = state->key[i+M] ^ (y>>1) ^ (-(y & 1) & MATRIX_A);
		}
		for (     ; i < N-1; i++)
		{
			y = (state->key[i] & UPPER_MASK) | (state->key[i+1] & LOWER_MASK);
			state->key[i] = state->key[i+(M-N)] ^ (y>>1) ^ (-(y & 1) & MATRIX_A);
		}
		y = (state->key[N - 1] & UPPER_MASK) | (state->key[0] & LOWER_MASK);
		state->key[N - 1] = state->key[M - 1] ^ (y >> 1) ^ (-(y & 1) & MATRIX_A);
		
		state->pos = 0;
	}
	y = state->key[state->pos++];
	
	/* Tempering */
	y ^= (y >> 11);
	y ^= (y << 7) & 0x9d2c5680UL;
	y ^= (y << 15) & 0xefc60000UL;
	y ^= (y >> 18);
	
	return y;
}

// Returns a random unsigned long between 0 and RK_MAX inclusive
double rk_double(rk_state * const state)
{
	/* shifts : 67108864 = 0x4000000, 9007199254740992 = 0x20000000000000 */
	long const a = rk_random(state) >> 5;
	long const b = rk_random(state) >> 6;
	return (a * 67108864.0 + b) / 9007199254740992.0;
}

// return two random Gaußian deviates with variance unity and zero mean.
void inline static rk_gauss(
		rk_state * const state,
		double * const x1,
		double * const x2,
		double const scale
		)
{
	double r2;
	
	do {
		// The order is intentional to ensure comparability to numpy.random.normal
		*x2 = 2.0*rk_double(state) - 1.0;
		*x1 = 2.0*rk_double(state) - 1.0;
		r2 = (*x1)*(*x1) + (*x2)*(*x2);
	}
	while (r2 >= 1.0 || r2 == 0.0);
	
	/* Box-Muller transform */
	double const f = scale*sqrt(-2.0*log(r2)/r2);
	*x1 *= f;
	*x2 *= f;
}

