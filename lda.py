import numpy
import random
import math
import pylab

# nCp[p[^
__alpha = 1.0
__beta = 1.0


def calc_lda_param( docs_dn, topics_dn, K, V ):
    D = len(docs_dn)

    n_dz = numpy.zeros((D,K))   # e¶dÉ¨¢ÄgsbNzª­¶µ½ñ
    n_zv = numpy.zeros((K,V))   # egsbNzÉÒ¨¢ÄPêvª­¶µ½ñ
    n_z = numpy.zeros(K)        # egsbNª­¶µ½ñ

    # ¦ã°é
    for d in range(D):
        N = len(docs_dn[d])    # ¶ÉÜÜêéPê
        for n in range(N):
            v = docs_dn[d][n]  # h
            z = topics_dn[d][n]     # PêÉèÄêêÄ¢égsbN
            n_dz[d][z] += 1
            n_zv[z][v] += 1
            n_z[z] += 1

    return n_dz, n_zv, n_z


def sample_topic( d, v, n_dz, n_zv, n_z, K, V ):
    P = [ 0.0 ] * K

    # ÝÏm¦ðvZ
    P = (n_dz[d,:] + __alpha )*(n_zv[:,v] + __beta) / (n_z[:] + V *__beta)
    for z in range(1,K):
        P[z] = P[z] + P[z-1]

    # TvO
    rnd = P[K-1] * random.random()
    for z in range(K):
        if P[z] >= rnd:
            return z



# PêðêñÉÀ×½XgÏ·
def conv_to_word_list( data ):
    V = len(data)
    doc = []
    for v in range(V):  # v:êbÌCfbNX
        for n in range(data[v]): # êbÌ­¶µ½ñ¶forðñ·
            doc.append(v)
    return doc

# ÞxvZ
def calc_liklihood( data, n_dz, n_zv, n_z, K, V  ):
    lik = 0

    # ãÌð¬»
    P_vz = (n_zv.T + __beta) / (n_z + V *__beta)
    for d in range(len(data)):
        Pz = (n_dz[d] + __alpha )/( numpy.sum(n_dz[d]) + K *__alpha )
        Pvz = Pz * P_vz
        Pv = numpy.sum( Pvz , 1 ) + 0.000001
        lik += numpy.sum( data[d] * numpy.log(Pv) )

    return lik

def save_model( n_dz, n_zv, n_z ):
    Pdz = n_dz + __alpha
    Pdz = (Pdz.T / Pdz.sum(1)).T

    Pzv = n_zv + __beta
    Pzv = (Pzv.T / Pzv.sum(1)).T

    numpy.savetxt( "Pdz.txt", Pdz, fmt=str("%f") )
    numpy.savetxt( "Pzv.txt", Pzv, fmt=str("%f") )


# ldaC
def lda( data , K ):
    pylab.ion()
    # ÞxÌXg
    liks = []

    # PêÌíÞ
    V = len(data[0])    # êb
    D = len(data)       # ¶

    # dataàÌPêðêñÉÀ×éivZµâ·­·é½ßj
    docs_dn = [ None for i in range(D) ]
    topics_dn = [ None for i in range(D) ]
    for d in range(D):
        docs_dn[d] = conv_to_word_list( data[d] )
        topics_dn[d] = numpy.random.randint( 0, K, len(docs_dn[d]) ) # ePêÉ_ÅgsbNðèÄé

    # LDAÌp[^ðvZ
    n_dz, n_zv, n_z = calc_lda_param( docs_dn, topics_dn, K, V )


    for it in range(20):
        # CÌ
        for d in range(D):
            N = len(docs_dn[d]) # ¶dÉÜÜêéPê
            for n in range(N):
                v = docs_dn[d][n]       # PêÌCfbNX
                z = topics_dn[d][n]     # PêÉèÄçêÄ¢égsbN


                # f[^ðæè«p[^ðXV
                n_dz[d][z] -= 1
                n_zv[z][v] -= 1
                n_z[z] -= 1

                # TvO
                z = sample_topic( d, v, n_dz, n_zv, n_z, K, V )

                # f[^ðTvO³ê½NXÉÇÁµÄp[^ðXV
                topics_dn[d][n] = z
                n_dz[d][z] += 1
                n_zv[z][v] += 1
                n_z[z] += 1

        lik = calc_liklihood( data, n_dz, n_zv, n_z, K, V )
        liks.append( lik )
        print ("対数尤度"), lik
        doc_dopics = numpy.argmax( n_dz , 1 )
        print ("分類結果"), doc_dopics
        print ("---------------------")


        # Ot\¦
        pylab.clf()
        pylab.subplot("121")
        pylab.title( "P(z|d)" )
        pylab.imshow( n_dz / numpy.tile(numpy.sum(n_dz,1).reshape(D,1),(1,K)) , interpolation="none" )
        pylab.subplot("122")
        pylab.title( "liklihood" )
        pylab.plot( range(len(liks)) , liks )
        pylab.draw()
        pylab.pause(0.1)

    save_model( n_dz, n_zv, n_z )
    pylab.ioff()
    pylab.show()

def main():
    data = numpy.loadtxt( "hist.txt" , dtype=numpy.int32)*10
    lda( data , 3 )

if __name__ == '__main__':
    main()

