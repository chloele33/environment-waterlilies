#version 300 es
precision highp float;

uniform vec3 u_Eye, u_Ref, u_Up;
uniform vec2 u_Dimensions;
uniform float u_Time;
uniform float u_RingSize;
uniform vec3 u_Color;
uniform float u_Blob;


in vec2 fs_Pos;
out vec4 out_Col;

float FOVY = radians(45.0);
float EPSILON = 0.01;
vec3 l_sha = normalize(vec3(1.0,0.8,-0.7));
vec3 lig = normalize(vec3(1.0,0.8,0.7));


float random1( vec3 p , vec3 seed) {
  return fract(sin(dot(p + seed, vec3(987.654, 123.456, 531.975))) * 85734.3545);
}

float random1( vec2 p , vec2 seed) {
  return fract(sin(dot(p + seed, vec2(127.1, 311.7))) * 43758.5453);
}

// combine height with fbm 3D
float interpNoise3D(float x, float y, float z) {
    float intX = floor(x);
    float intY = floor(y);
    float intZ = floor(z);
    float fractX = fract(x);
    float fractY = fract(y);
    float fractZ = fract(z);

    float v1 = random1(vec3(intX, intY, intZ), vec3(1.0, 1.0, 1.0));
    float v2 = random1(vec3(intX, intY, intZ + 1.0), vec3(1.0, 1.0, 1.0));
    float v3 = random1(vec3(intX + 1.0, intY, intZ + 1.0), vec3(1.0, 1.0, 1.0));
    float v4 = random1(vec3(intX + 1.0, intY, intZ), vec3(1.0, 1.0, 1.0));
    float v5 = random1(vec3(intX, intY + 1.0, intZ), vec3(1.0, 1.0, 1.0));
    float v6 = random1(vec3(intX, intY + 1.0, intZ + 1.0), vec3(1.0, 1.0, 1.0));
    float v7 = random1(vec3(intX + 1.0, intY + 1.0, intZ), vec3(1.0, 1.0, 1.0));
    float v8 = random1(vec3(intX + 1.0, intY + 1.0, intZ + 1.0), vec3(1.0, 1.0, 1.0));


    float i1 = mix(v1, v2, fractX);
    float i2 = mix(v3, v4, fractX);
    float i3 = mix(v5, v6, fractX);
    float i4 = mix(v7, v8, fractX);

    float i5 = mix(i1, i2, fractY);
    float i6 = mix(i3, i4, fractY);

    return mix(i5, i6, fractZ);
}

float fbm3d(vec3 pos) {
  float total = 0.f;
  float persistence = 0.5f;
  int octaves = 15;

  //vec3 pos = vec3(x, y, z);

  for (int i = 0; i < octaves; i++) {
    float freq = pow(2.0, float(i));
    float amp = pow(persistence, float(i));
    total += abs(interpNoise3D( pos.x / 80.0  * freq, pos.y / 10.0 * freq, pos.z / 20.0 * freq)) * amp;
  }
  return  total;
}


float interpNoise2D(float x, float y) {
    float intX = floor(x);
    float intY = floor(y);
    float fractX = fract(x);
    float fractY = fract(y);

    float v1 = random1(vec2(intX, intY), vec2(1.0, 1.0));
    float v2 = random1(vec2(intX + 1.0, intY), vec2(1.0, 1.0));
    float v3 = random1(vec2(intX, intY + 1.0), vec2(1.0, 1.0));
    float v4 = random1(vec2(intX + 1.0, intY + 1.0), vec2(1.0, 1.0));

    float i1 = mix(v1, v2, fractX);
    float i2 = mix(v3, v4, fractX);
    return mix(i1, i2, fractY);
}

float fbm(float x, float y) {
  float total = 0.f;
  float persistence = 0.5f;
  int octaves = 8;
  float roughness = 1.0;

  vec2 pos = vec2(x, y);
  vec2 shift = vec2(100.0);

  mat2 rot = mat2(cos(0.5), sin(0.5),
                      -sin(0.5), cos(0.50));

  for (int i = 0; i < octaves; i++) {
    float freq = pow(2.0, float(i));
    float amp = pow(persistence, float(i));

    pos = rot * pos * 1.0 + shift;

    total += abs(interpNoise2D( pos.x / 100.0  * freq * sin(u_Time * 0.1), pos.y / 200.0 * freq)) * amp * roughness;
    roughness *= interpNoise2D(pos.x / 5.0  * freq, pos.y / 5.0 * freq);
  }
  return  total;
}

float worley(float x, float y, float scale) {
    float scale_invert = abs(80.0 - scale);
    vec2 pos = vec2(x/scale_invert, y/scale_invert);

    float m_dist = 40.f;  // minimun distance
    vec2 m_point = vec2(0.f, 0.f);       // minimum point

    for (int j=-1; j<=1; j++ ) {
        for (int i=-1; i<=1; i++ ) {
            vec2 neighbor = vec2(floor(pos.x) + float(j), floor(pos.y) + float(i));
            vec2 point = neighbor + random1(neighbor, vec2(1.f, 1.f));
            float dist = distance(pos, point);

            if( dist < m_dist ) {
                m_dist = dist;
                m_point = point;
            }
        }
    }
    return m_dist;
}


// reference from class slides
float triangleWave(float x, float freq, float amp) {
    float tri = floor(abs(mod(floor(x*freq), amp) - (0.5 * amp)));
    return tri;
}

// a function that uses the NDC coordinates of the current fragment (i.e. its fs_Pos value) and projects a ray from that pixel.
vec3 castRay(vec3 eye) {
    float len = length(u_Ref - eye);
    vec3 F = normalize(u_Ref - eye);
    vec3 R = normalize(cross(F, u_Up));
    float aspect = u_Dimensions.x / u_Dimensions.y;
    float alpha = FOVY / 2.0;
    vec3 V = u_Up * len * tan(alpha);
    vec3 H = R * len * aspect * tan(alpha);

    vec3 point = u_Ref + (fs_Pos.x * H + fs_Pos.y * V);
    vec3 ray_dir = normalize(point - eye);


    return ray_dir;
}

float opSmoothIntersection( float d1, float d2, float k )
{
    float h = clamp( 0.5 - 0.5*(d2-d1)/k, 0.0, 1.0 );
    return mix( d2, d1, h ) + k*h*(1.0-h);
    }


float sdEllipsoid(vec3 p, vec3 r )
{
    float k0 = length(p/r);
    float k1 = length(p/(r*r));
    return k0*(k0-1.0)/k1;
}
//rotate
mat2 rot(float a){
	return mat2(cos(a), -sin(a), sin(a), cos(a));
}

//Sphere SDF
float sphereSDF(vec3 p, float r, vec3 scale){
  return length(p * scale) - r;
}

//Box SDF
float sdBox( vec3 p, vec3 b )
{
  vec3 d = abs(p) - b;
  return length(max(d,0.0))+ min(max(d.x,max(d.y,d.z)),0.0); // remove this line for an only partially signed sdf
}

float sdCylinder( vec3 p, float ra, float rb, float h )
{

    vec2 d = vec2( length(p.xz)-ra+rb, abs(p.y) - h );
    return min(max(d.x,d.y),0.0) + length(max(d,0.0)) - rb;
}

// union
float opUnion(float d1, float d2) {
    return min(d1, d2);
}

float opSubtraction( float d1, float d2 ) { return max(d1,-d2); }

float opIntersection( float d1, float d2 ) { return max(d1,d2); }


float leaf(vec3 p, float r, float h, float rotate)
{
    //return opSubtraction(sdBox(p, vec3(r, h, r)), sdCylinder(p - vec3(r, 0.0, r), r*2.0, 0.0, h)) ;
    return opSubtraction(sdCylinder(p, r , 0.0, h), sdBox(p - vec3(r * rotate, 0.0, -r * rotate), vec3(r/2.0, 1.0, r/2.0))) ;
}


//Torus SDF
float torusSDF( vec3 p, vec2 t )
{
  vec2 q = vec2(length(p.xz)-t.x,p.y);
  return length(q)-t.y;
}

//round box SDF
float roundBoxSDF( vec3 p, vec3 b, float r )
{
  vec3 d = abs(p) - b;
  return length(max(d,0.0)) - r
         + min(max(d.x,max(d.y,d.z)),0.0); // remove this line for an only partially signed sdf
}



//Flower petal
float flowerPetal(vec3 pos, float rad, float numPetals, float curve, float rotation)
{
    pos.yz = rot(-0.05) * pos.yz;
    pos.xz = rot(rotation + 0.3 * sin(0.5 ) ) * pos.xz;
	pos.y += 0.6;

	//repeat
//    vec2 q = floor((pos.xz - 0.75) / 1.5);
//    pos.xz = mod(pos.xz - 0.75, 1.5) - .75;
    float angle = atan(pos.z, pos.x);
    float radius = length(pos.xz) * rad;

    //set up repeat petals
    float div = 3.14 / numPetals * 2.0;
    float a = mod(angle, div) - 0.6 * div;
    //pointy petal
    float r = radius + 0.25*abs(a);
    vec3 p = pos;
    p.x = r* cos(a) * rad;
    p.z = r * sin(a)* rad * 6.0;
    // curl and then flatten flower
    p.xy = rot(curve * max(curve - 0.1 * curve, abs(sin(u_Time * 0.02)) + 0.15) - cos(2. * sin(0.75+ 1.0 * (p.x + 0.5 * p.y))) * .75 * r) * p.xy;
    p.y = p.y * 10.0;
    return 0.07 * sphereSDF(p + vec3(-0.3, 0., 0.), rad, vec3(1.0, 1.0, 1.0));
}

float waterLily(vec3 p) {
    float flower = flowerPetal(p, 0.6, 9.0, 0.3, -0.6);
    float flower1 = flowerPetal(p + vec3(0.0, -0.05, 0.0), 0.6, 8.0, 0.7, -0.2);
    float layer1 = opUnion(flower, flower1);
    float flower2 = flowerPetal(p + vec3(0.0, -0.08, 0.0), 0.55, 8.0, 1.0, 0.0);
    float layer2 = opUnion(layer1, flower2);
    float flower3 = flowerPetal(p + vec3(0.0, -0.09, 0.0), 0.50, 5.0, 1.2, 0.0);

    return opUnion(layer2, flower3);
}

float waterLily2(vec3 p) {
    //float flower = flowerPetal(p, 0.55, 8.0, 0.5, -0.6);
    float flower1 = flowerPetal(p + vec3(0.0, -0.05, 0.0), 0.55, 5.0, 0.8, -0.2);
    //float layer1 = opUnion(flower, flower1);
    float flower2 = flowerPetal(p + vec3(0.0, -0.08, 0.0), 0.50, 6.0, 1.1, 0.0);
    float layer2 = opUnion(flower1, flower2);
    float flower3 = flowerPetal(p + vec3(0.0, -0.09, 0.0), 0.45, 4.0, 1.2, 0.0);

    return opUnion(layer2, flower3);
}


float allGreens(vec3 p) {
        float leaf1 = leaf(p+ vec3(3.0, -0.2, 2.0), 1.5, 0.01, 1.0);
        float leaf2 = leaf(p+ vec3(3.0, -0.2, -3.0), 1.0, 0.01, -1.0);
        float leaf3 = leaf(p+ vec3(-2.0, -0.2, -10.0), 1.5, 0.01, 1.0);
        float leafSet1 = opUnion(leaf1, leaf2);

        return opUnion(leafSet1, leaf3);

}

//sceneSDF
float sceneSDF(vec3 p, in vec3 color)
{
    float lily1 = waterLily(p + vec3(-1.0, 0.0, -1.0));
    //float lily2 = waterLily2(p + vec3(8.0, 0.0, -5.0));

    //return opUnion(lily1, lily2);
    return lily1;
}

float greenSDF(vec3 p) {
    return allGreens(p);
}

// Water
// intersection functions
float intersectPlane(vec3 origin, vec3 dir, vec4 n) {
	float t = -(dot(origin, n.xyz) + n.w) / dot(dir, n.xyz);
	return t;
}

float waterMap( vec2 pos ) {
	vec2 posm = pos * mat2( 0.60, -0.80, 0.80, 0.60 );
	vec3 PosVec = vec3( 8.*posm, u_Time );
	return abs( fbm(PosVec.x, PosVec.y)-0.5 )* 0.1;
}


vec3 waterNormal(vec3 origin, float dist, vec3 n) {
    vec2 coord = origin.xz;
    vec3 normal = vec3( 0., 1., 0. );
    // have a bump offset to make noise dependent on dist
    float bump = 0.1 * (1. - smoothstep( 0., 100.0, dist));
    vec2 dx = vec2(0.1, 0.0);
    vec2 dz = vec2(0.0, 0.1);
    normal.x = -bump * (waterMap(coord + dx) - waterMap(coord-dx) ) / (2. * EPSILON);
    normal.z = -bump * (waterMap(coord + dz) - waterMap(coord-dz) ) / (2. * EPSILON);
    normal = normalize( normal );

    return normal;
}


// flog clouds

#define CLOUDSCALE (500./(64.*0.03))

float cloudMap( const in vec3 p, const in float ani ) {
	vec3 r = p/CLOUDSCALE;

	float den = -1.8+cos(r.y*5.-4.3);

	float f;
	vec3 q = 2.5*r*vec3(0.75,1.0,0.75)  + vec3(1.0,1.0,15.0)*ani*0.15;
    f  = 0.50000*fbm3d( q ); q = q*2.02 - vec3(-1.0,1.0,-1.0)*ani*0.15;
    f += 0.25000*fbm3d( q ); q = q*2.03 + vec3(1.0,-1.0,1.0)*ani*0.15;
    f += 0.12500*fbm3d( q ); q = q*2.01 - vec3(1.0,1.0,-1.0)*ani*0.15;
    f += 0.06250*fbm3d( q ); q = q*2.02 + vec3(1.0,1.0,1.0)*ani*0.15;
    f += 0.03125*fbm3d( q );

	return 0.065*clamp( den + 4.4*f, 0.0, 1.0 );
}

vec3 raymarchClouds( const in vec3 ro, const in vec3 rd, const in vec3 bgc, const in vec3 fgc, const in float startdist, const in float maxdist, const in float ani ) {
    // dithering
    float h = fract(sin(rd.x+35.6987221*rd.y+u_Time)*43758.5453);
	float t = startdist+CLOUDSCALE*0.02*h;//0.1*texture( iChannel0, fragCoord.xy/iChannelResolution[0].x ).x;

    // raymarch
	vec4 sum = vec4( 0.0 );
	for( int i=0; i<64; i++ ) {
		if( sum.a > 0.99 || t > maxdist ) continue;

		vec3 pos = ro + t*rd;
		float a = cloudMap( pos, ani );

        // lighting
		float dif = clamp(0.1 + 0.8*(a - cloudMap( pos + lig*0.15*CLOUDSCALE, ani )), 0., 0.5);
		vec4 col = vec4( (1.+dif)*fgc, a );
		// fog
	//	col.xyz = mix( col.xyz, fgc, 1.0-exp(-0.0000005*t*t) );
		col.rgb *= col.a;
		sum = sum + col*(1.0 - sum.a);

        // advance ray with LOD
		t += (0.03*CLOUDSCALE)+t*0.012;
	}
    // blend with background
	sum.xyz = mix( bgc, sum.xyz/(sum.w+0.0001), sum.w );

	return clamp( sum.xyz, 0.0, 1.0 );
}


// Bounding Volumne Hierarchy
// Cube
struct Cube {
	vec3 min;
	vec3 max;
};

// Ray-cube intersection
float cubeIntersect(vec3 raydir, vec3 origin, Cube cube) {
    float tNear = -9999999.0;
    float tFar = 9999999.0;
    float far = 9999999.0;
    //X SLAB

    //if ray is parallel to x plane
    if (raydir.x == 0.0f) {
        if (origin.x < cube.min.x) {
            return far;
        }
        if (origin.x > cube.max.x) {
            return far;
        }
    }
    float t0 = (cube.min.x - origin.x) / raydir.x;
    float t1 = (cube.max.x - origin.x) / raydir.x;
    // swap
    if (t0 > t1) {
        float temp = t0;
        t0 = t1;
        t1 = temp;
    }
    if (t0 > tNear) {
        tNear = t0;
    }
    if (t1 < tFar) {
        tFar = t1;
    }

    //Y SLAB
    if (raydir.y == 0.0f) {
        if (origin.y < cube.min.y) {
            return far;
        }
        if (origin.y > cube.max.y) {
            return far;
        }
    }
    t0 = (cube.min.y - origin.y) / raydir.y;
    t1 = (cube.max.y - origin.y) / raydir.y;
    // swap
    if (t0 > t1) {
        float temp = t0;
        t0 = t1;
        t1 = temp;
    }
    if (t0 > tNear) {
        tNear = t0;
    }
    if (t1 < tFar) {
        tFar = t1;
    }

     //Z SLAB
    if (raydir.z == 0.0f) {
        if (origin.z < cube.min.z) {
            return far;
        }
        if (origin.z > cube.max.z) {
            return far;
        }
    }
    t0 = (cube.min.z - origin.z) / raydir.z;
    t1 = (cube.max.z - origin.z) / raydir.z;
    // swap
    if (t0 > t1) {
        float temp = t0;
        t0 = t1;
        t1 = temp;
    }
    if (t0 > tNear) {
        tNear = t0;
    }
    if (t1 < tFar) {
        tFar = t1;
    }

//    if (tNear < 0.0) {
//        return far;
//    }

    // missed the cube
    if (tNear > tFar) {
        return far;
    }

    return tNear;

}


Cube sceneBB() {
	Cube cube;
	cube.min = vec3(-7.0, -8.0, -7.0);
	cube.max = vec3(7.0, 8.0, 7.0);
	return cube;
}


float BVH(vec3 origin, vec3 dir, Cube cubes[5]) {
    float currT = 999999.0;
    for (int i = 0; i < cubes.length(); i++) {
        float t = cubeIntersect(dir, origin, cubes[i]);
        if (currT > t) {
            currT = t;
        }
    }
    return currT;
  }

float softshadow( vec3 ro, vec3 rd, float mint, float maxt, float k)
{
    float res = 1.0;
    for( float t = mint; t < maxt;)
    {
        float h = sceneSDF(ro + rd*t,vec3(1.0, 1.0, 1.0) );
        if( h<0.001 )
            return 0.0;
        res = min( res, k*h/t );
        t += h;
    }
    return res;
}


float rayMarch(vec3 rayDir, vec3 cameraOrigin, in vec3 color)
{
    // check for bounding boxes
//    Cube cubes[5];
//    cubes[0] = sceneBB();
//    cubes[1] = sphereBB();
//    cubes[2] = metaCubesBB();
//    cubes[3] = torusBB();
//    cubes[4] = flatCynBB();
//    if (BVH(cameraOrigin, rayDir, cubes) > 50.0) {
//        return 10000.0;
//    }
    int INTERATIONS = 150;
	float MAX_DIST = 50.0;

    float totalDist = 0.0;
	vec3 pos = cameraOrigin;
	float dist = 0.0;

    for(int i = 0; i < INTERATIONS; i++)
	{
		dist = sceneSDF(pos, vec3(1.0, 1.0, 1.0));
		totalDist = totalDist + dist;
		pos += dist * rayDir;

        if(dist < EPSILON || totalDist > MAX_DIST)
		{
			break;
		}
	}

    return totalDist  ;
}

float rayMarchGreen(vec3 rayDir, vec3 cameraOrigin)
{
    // check for bounding boxes
//    Cube cubes[5];
//    cubes[0] = sceneBB();
//    cubes[1] = sphereBB();
//    cubes[2] = metaCubesBB();
//    cubes[3] = torusBB();
//    cubes[4] = flatCynBB();
//    if (BVH(cameraOrigin, rayDir, cubes) > 50.0) {
//        return 10000.0;
//    }
    int INTERATIONS = 150;
	float MAX_DIST = 50.0;

    float totalDist = 0.0;
	vec3 pos = cameraOrigin;
	float dist = 0.0;

    for(int i = 0; i < INTERATIONS; i++)
	{
		dist = greenSDF(pos);
		totalDist = totalDist + dist;
		pos += dist * rayDir;

        if(dist < EPSILON || totalDist > MAX_DIST)
		{
			break;
		}
	}

    return totalDist  ;
}

// calculate normals
vec3 getNormal(vec3 p) {
    float eps = 0.0001;
     vec2 h = vec2(eps,0);
     vec3 color;
    return normalize( vec3(sceneSDF(p+h.xyy, color) - sceneSDF(p-h.xyy, color),
                           sceneSDF(p+h.yxy, color) - sceneSDF(p-h.yxy, color),
                           sceneSDF(p+h.yyx, color) - sceneSDF(p-h.yyx, color) ) );
}

vec3 getNormalGreen(vec3 p) {
    float eps = 0.0001;
     vec2 h = vec2(eps,0);
    return normalize( vec3(greenSDF(p+h.xyy) - greenSDF(p-h.xyy),
                           greenSDF(p+h.yxy) - greenSDF(p-h.yxy),
                           greenSDF(p+h.yyx) - greenSDF(p-h.yyx) ) );
}






vec3 getShading(vec3 pos , vec3 lightp, vec3 color, vec3 rayDir, bool shadow)
{
	vec3 norm = getNormal(pos);
    vec3 lightdir = normalize(pos - lightp);

    float sha = 0.0;
    float dif = max(dot(norm,l_sha),0.0);
    if( shadow == true && dif > 0.01 ) {
        sha = softshadow( pos+0.001 * norm, l_sha, 0.005, 50.0, 64.0 );
    }

    vec3 amb = vec3(0.05);
    vec3 diffuse = vec3(0.5 * pow(0.5+0.5*dot(norm, -lightdir), 3.0));
    vec3 phong = vec3(0.2 * pow(max(dot(-rayDir, reflect(lightdir, norm)), 0.0), 20.0));

    return (amb + diffuse + phong) * color * pow(vec3(sha),vec3(1.0,1.0,1.5));
}

vec3 getShadingGreen(vec3 pos , vec3 lightp, vec3 color, vec3 rayDir, bool shadow)
{
	vec3 norm = getNormalGreen(pos);
    vec3 lightdir = normalize(pos - lightp);

    float sha = 0.0;
    float dif = max(dot(norm,l_sha),0.0);
    if( shadow == true && dif > 0.01 ) {
        sha = softshadow( pos+0.001 * norm, l_sha, 0.005, 50.0, 64.0 );
    }

    vec3 amb = vec3(0.3);
    vec3 diffuse = vec3(0.5 * pow(0.5+0.5*dot(norm, -lightdir), 3.0));
    //vec3 phong = vec3(0.8 * pow(max(dot(-rayDir, reflect(lightdir, norm)), 0.0), 20.0));

    return (amb + diffuse ) * color * pow(vec3(sha),vec3(1.0,1.0,1.5));
}


vec3 backgroundColor(vec3 dir ) {
	float sun = clamp(dot(lig, dir), 0.0, 1.0 );
	vec3 col = vec3(0.7, 0.69, 0.75) - dir.y*0.2*vec3(1.0,0.8,1.0) + 0.15*0.75;
	col += vec3(1.0,.6,0.1)*pow( sun, 8.0 );
	col *= 0.95;
	return col;
}


void main() {

  vec3 dir = castRay(u_Eye);

  //vec3 color = 0.5 * (dir + vec3(1.0, 1.0, 1.0));
  //out_Col = vec4(0.5 * (vec2(1.0)), 0.5 * (sin(u_Time * 3.14159 * 0.01) + 1.0), 1.0);

  vec3 color  = backgroundColor(dir);
  vec3 backgroundCol = color;
   // water
   float fresnel;
   vec4 waterPlaneNormal = vec4(0, 1.0, 0, 0);
   float dist = intersectPlane( u_Eye, dir,  waterPlaneNormal);
   vec3 reflection = vec3(dir);
   bool hitwater = false;
    if (dist > 0.0) {
       vec3 p = u_Eye + dist * dir;
       hitwater = true;
       vec3 waterNorm = waterNormal(p, dist * 5.0, vec3(0, 1.0, 0));
       reflection = reflect(dir, waterNorm);
       float nrdot = dot(waterNorm,dir);
       fresnel = pow(1.0-abs(nrdot),5.);
        backgroundCol = backgroundColor(dir);
        color = backgroundCol;
    }
     //color = raymarchClouds( u_Eye, dir, color, backgroundCol, hitwater?max(0.,min(150.,(150.-dist))):150., 500.0, u_Time*0.05 );
     if( hitwater ) {
     		color = mix( color.xyz, backgroundCol, 1.0-exp(-0.0000005*dist*dist) );
     		color *= fresnel * 0.95;
     	}
    float t = rayMarch(reflection, u_Eye + dist * dir, color);

  if (t < 50.0){
    vec3 light1 = getShading(u_Eye + t * dir, vec3(5.0,10.0,-20.0), vec3(1.0,1.0,1.0), dir, true);
    vec3 light2 = getShading(u_Eye + t * dir, vec3(-20,10.0,5.0), vec3(0.5,0.4,0.1), dir, false);
    vec3 light3 = getShading(u_Eye + t * dir, vec3(20.0,5.0,-8.0), vec3(0.7,0.3,0.1), dir, false);
    color = light1+light2+light3;
  }

  color = pow( color, vec3(0.7) );


        // load Scene SDF
  t = rayMarch(dir, u_Eye, color);

  if (t < 50.0){
    vec3 point = u_Eye + t * dir;
    vec3 light1 = getShading(u_Eye + t * dir, vec3(5.0,10.0,-20.0), vec3(1.0,1.0,1.0), dir, true);
    vec3 light4 = getShading(u_Eye + t * dir, vec3(-5.0,15.0,20.0), vec3(1.0,1.0,1.0), dir, true);
    vec3 light2 = getShading(u_Eye + t * dir, vec3(-20,10.0,5.0), vec3(0.8,0.5,0.5), dir, false);
    vec3 light3 = getShading(u_Eye + t * dir, vec3(20.0,5.0,-8.0), vec3(0.7,0.3,0.3), dir, false);
    color = light1+light2+light3 + light4;
    color *= 1.5;
    //out_Col = vec4(color , 1.0);

  } else {

        float g = rayMarchGreen(dir, u_Eye);
        if (g < 50.0){
            vec3 point = u_Eye + g * dir;
            vec3 light1 = getShadingGreen(u_Eye + g * dir, vec3(5.0,10.0,-10.0), vec3(0.6,0.78,0.5), dir, true);
            vec3 light4 = getShadingGreen(u_Eye + g * dir, vec3(-5.0,15.0,10.0), vec3(0.9,0.9,0.9), dir, false);
            vec3 light2 = getShadingGreen(u_Eye + g * dir, vec3(-20,10.0,5.0), vec3(0.5,0.4,0.1), dir, false);
            vec3 light3 = getShadingGreen(u_Eye + g * dir, vec3(20.0,5.0,-8.0), vec3(0.7,0.3,0.1), dir, false);
            color = light1+light2 + light3 + light4;
            color *= 1.2;

             vec3 highlight = light1;
             float textureMap = worley(point.x * 80.0, point.z * 80.0 ,120.0) - 0.15 * fbm(fs_Pos.x, fs_Pos.y);

             color = textureMap * (highlight) + (1.0 - textureMap) * (color);


        }
    }
    // distance fog
    float fog = clamp(smoothstep(15.0, 50.0, length(fs_Pos)), 0.0, 1.0); // Distance fog

  	// contrast
  	color = color*color*(3.0-2.0*color);

  	// saturation
    color = mix( color, vec3(dot(color,vec3(0.50))), -0.5 );

    //VIGNETTE
    float fallOff = 0.25;
    vec2 uv = gl_FragCoord.xy / u_Dimensions.xy;
   	//color *= 0.25 + 0.75*pow( 16.0 * uv.x * uv.y * (1.0-uv.x)*(1.0-uv.y), 0.1 );
   	vec2 coord = (uv - 0.5) * (u_Dimensions.x/u_Dimensions.y) * 2.0;
    float rf = sqrt(dot(coord, coord)) * fallOff;
    float rf2_1 = rf * rf + 1.0;
    float e = 1.0 / (rf2_1 * rf2_1);
    color *= e;

    out_Col = vec4(mix(color, vec3(205.0 / 255.0, 233.0 / 255.0, 1.0) ,fog), 1.0 );

}
