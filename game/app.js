// ============================================================
// GPU Architect: Rise from Silicon — Game Engine + UI
// ============================================================

// === STATE ===
const DEFAULT_STATE = {
  name:'',fc:0,sc:0,fcTotal:0,scTotal:0,
  rankIdx:0,completed:[],masteryPassed:[],bossCleared:[],
  quizScores:{},quizAttempts:{},codeTracesDone:{},reviewsDone:{},
  maxCombo:0,gflops:0,perfectModules:0,
  streak:0,longestStreak:0,lastDate:null,
  inventory:{},earnedBadges:[],
  hintsUsed:{},
};
let S = loadState();
function loadState(){ try{return JSON.parse(localStorage.getItem('gpu_architect_save'))||{...DEFAULT_STATE}}catch(e){return{...DEFAULT_STATE}} }
function saveState(){ localStorage.setItem('gpu_architect_save',JSON.stringify(S)); }
function addFC(n){ S.fc+=n; S.fcTotal+=n; saveState(); updateNav(); }
function addSC(n){ S.sc+=n; saveState(); updateNav(); }
function spendFC(n){ if(S.fc>=n){S.fc-=n;saveState();updateNav();return true}return false; }
function spendSC(n){ if(S.sc>=n){S.sc-=n;saveState();updateNav();return true}return false; }

// === ENGINES ===
function getComboMult(c){ return c>=7?3:c>=5?2:c>=3?1.5:1; }
function getRank(){ return RANKS[S.rankIdx]; }
function calcGflops(){
  const r=getRank();
  const scores=Object.values(S.quizScores);
  const avg=scores.length?scores.reduce((a,b)=>a+b,0)/scores.length/100:0;
  S.gflops=Math.floor(r.gflops*(0.5+avg*0.5));
  saveState();
}
function checkRankUp(){
  let newIdx=S.rankIdx;
  for(let i=S.rankIdx+1;i<RANKS.length;i++){
    const r=RANKS[i];
    const modsOk=r.modules.every(m=>S.completed.includes(m));
    const mastOk=!r.masteryReq||S.masteryPassed.includes(r.masteryReq);
    if(modsOk&&mastOk) newIdx=i; else break;
  }
  if(newIdx>S.rankIdx){
    S.rankIdx=newIdx;
    calcGflops();
    saveState();
    showToast(`승진! ${RANKS[newIdx].icon} ${RANKS[newIdx].name}`,'rank');
    return true;
  }
  return false;
}
function checkAchievements(){
  const info={completed:S.completed,bossCleared:S.bossCleared,maxCombo:S.maxCombo,
    gflops:S.gflops,fcTotal:S.fcTotal,perfectModules:S.perfectModules,
    quizScores:S.quizScores,codeTracesDone:S.codeTracesDone};
  ACHIEVEMENTS.forEach(a=>{
    if(!S.earnedBadges.includes(a.id)&&a.cond(info)){
      S.earnedBadges.push(a.id);
      if(a.sc)addSC(a.sc);
      showToast(`배지 획득! ${a.icon} ${a.name}`,'badge');
    }
  });
  saveState();
}
function isModuleAvailable(id){
  const idx=MODULES.findIndex(m=>m.id===id);
  if(idx===0)return true;
  return S.completed.includes(MODULES[idx-1].id);
}
function checkStreak(){
  const today=new Date().toDateString();
  if(S.lastDate===today)return;
  const yesterday=new Date(Date.now()-86400000).toDateString();
  if(S.lastDate===yesterday){S.streak++;} else if(S.lastDate!==today){S.streak=1;}
  S.lastDate=today;
  if(S.streak>S.longestStreak)S.longestStreak=S.streak;
  const bonus=Math.min(S.streak*10,100);
  addFC(bonus);
  showToast(`일일 접속 ${S.streak}일! +${bonus} FC`,'reward');
  saveState();
}

// === TOASTS ===
let toastTimeout;
function showToast(msg,type='reward'){
  const old=document.querySelector('.toast');if(old)old.remove();
  const t=document.createElement('div');t.className=`toast ${type}`;t.textContent=msg;
  document.body.appendChild(t);
  clearTimeout(toastTimeout);
  toastTimeout=setTimeout(()=>t.remove(),3000);
}

// === NAV ===
function updateNav(){
  const nav=document.getElementById('topnav');if(!nav)return;
  const r=getRank();
  nav.innerHTML=`
    <div class="logo" onclick="navigate('dashboard')" style="cursor:pointer">${r.icon} GPU Architect</div>
    <div class="stats">
      <span>${r.name}</span>
      <span class="fc-icon">⚡${S.fc}</span>
      <span class="sc-icon">💎${S.sc}</span>
      <span style="color:var(--green)">${S.gflops.toLocaleString()} GFLOP/s</span>
    </div>`;
}

// === NAVIGATION ===
let currentScreen='';
let screenParams={};
function navigate(screen,params={}){
  currentScreen=screen;screenParams=params;
  const main=document.getElementById('main');
  main.innerHTML='';
  switch(screen){
    case'welcome':renderWelcome(main);break;
    case'dashboard':renderDashboard(main);break;
    case'module':renderModule(main,params.id);break;
    case'quiz':renderQuiz(main,params.id);break;
    case'codetrace':renderCodeTrace(main,params.id);break;
    case'review':renderReview(main,params.id);break;
    case'boss':renderBoss(main,params.id);break;
    case'shop':renderShop(main);break;
    case'badges':renderBadges(main);break;
    case'result':renderResult(main,params);break;
    default:renderDashboard(main);
  }
  window.scrollTo(0,0);
}

// === SCREENS ===

function renderWelcome(el){
  el.innerHTML=`
    <div style="text-align:center;padding:60px 20px">
      <div style="font-size:3rem;margin-bottom:16px">🏗️</div>
      <h1 style="font-size:1.8rem;margin-bottom:8px;background:linear-gradient(90deg,var(--cyan),var(--gold));-webkit-background-clip:text;-webkit-text-fill-color:transparent">
        GPU Architect:<br>Rise from Silicon</h1>
      <p style="color:var(--dim);margin-bottom:32px">SiliconForge에서 가장 빠른 행렬곱 엔진을 만드세요<br>나이브 50 GFLOP/s → cuBLAS 95% (9,500 GFLOP/s)</p>
      <input id="nameInput" placeholder="이름을 입력하세요" style="padding:12px 20px;border-radius:8px;border:2px solid var(--border);background:var(--card);color:var(--text);font-size:1rem;width:240px;margin-bottom:16px;text-align:center">
      <br>
      <button class="btn btn-primary" onclick="startGame()" style="font-size:1.1rem;padding:14px 40px">입사하기</button>
      <p style="color:var(--dim);margin-top:24px;font-size:.8rem">75개 퀴즈 · 6개 코드 추적 · 3개 보스전 · 33개 배지</p>
    </div>`;
  setTimeout(()=>{const inp=document.getElementById('nameInput');if(inp)inp.focus()},100);
}
function startGame(){
  const name=document.getElementById('nameInput').value.trim();
  if(!name)return showToast('이름을 입력해주세요','error');
  S.name=name;saveState();
  document.getElementById('topnav').style.display='flex';
  updateNav();checkStreak();navigate('dashboard');
}

function renderDashboard(el){
  const r=getRank();
  const nextR=S.rankIdx<RANKS.length-1?RANKS[S.rankIdx+1]:null;
  const completedPct=Math.round(S.completed.length/11*100);
  el.innerHTML=`
    <div class="rank-display">
      <div class="rank-icon">${r.icon}</div>
      <div class="rank-info">
        <div class="rank-name">${r.name}</div>
        <div class="rank-sub">${r.en} · ${S.name}</div>
        <div class="gflops">${S.gflops.toLocaleString()} GFLOP/s · cuBLAS ${r.cubl}%</div>
      </div>
    </div>
    ${nextR?`<div class="card" style="padding:12px 16px">
      <div style="display:flex;justify-content:space-between;font-size:.85rem">
        <span>다음 직급: ${nextR.icon} ${nextR.name}</span>
        <span style="color:var(--dim)">필요: ${nextR.modules.filter(m=>!S.completed.includes(m)).map(m=>'모듈 '+m).join(', ')||'준비 완료'}${nextR.masteryReq&&!S.masteryPassed.includes(nextR.masteryReq)?' + 마스터리 '+nextR.masteryReq:''}</span>
      </div>
    </div>`:''}
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
        <h2 style="margin:0">학습 모듈</h2>
        <span style="color:var(--dim);font-size:.85rem">${S.completed.length}/11 완료 (${completedPct}%)</span>
      </div>
      <div class="progress-bar"><div class="progress-fill cyan" style="width:${completedPct}%"></div></div>
      <div class="module-grid" id="moduleGrid"></div>
    </div>
    <div style="display:flex;gap:10px;flex-wrap:wrap">
      <button class="btn btn-secondary" onclick="navigate('shop')">🛒 상점</button>
      <button class="btn btn-secondary" onclick="navigate('badges')">🏆 배지 (${S.earnedBadges.length}/${ACHIEVEMENTS.filter(a=>!a.hidden||S.earnedBadges.includes(a.id)).length})</button>
      <button class="btn btn-secondary" onclick="if(confirm('진행 데이터를 초기화하시겠습니까?')){localStorage.removeItem('gpu_architect_save');location.reload()}">초기화</button>
    </div>
    <div class="card" style="margin-top:16px">
      <h3>통계</h3>
      <div class="stat-row">
        <div class="stat-box"><div class="stat-val">${S.streak}</div><div class="stat-label">연속 접속</div></div>
        <div class="stat-box"><div class="stat-val">${S.maxCombo}</div><div class="stat-label">최대 콤보</div></div>
        <div class="stat-box"><div class="stat-val">⚡${S.fcTotal}</div><div class="stat-label">총 FC 획득</div></div>
        <div class="stat-box"><div class="stat-val">${Object.keys(S.quizScores).length}</div><div class="stat-label">퀴즈 완료</div></div>
      </div>
    </div>`;
  const grid=document.getElementById('moduleGrid');
  MODULES.forEach(m=>{
    const done=S.completed.includes(m.id);
    const avail=isModuleAvailable(m.id);
    const isMastery=m.mastery;
    const score=S.quizScores[m.id];
    const cell=document.createElement('div');
    cell.className=`module-cell ${done?'completed':avail?'current':'locked'} ${isMastery?'mastery':''}`;
    cell.innerHTML=`
      <div class="mod-num">${m.id}</div>
      <div class="mod-title">${m.sub}</div>
      ${score!=null?`<div class="mod-score">${score}%</div>`:''}
      ${!avail?'<div class="lock-icon">🔒</div>':''}`;
    if(avail)cell.onclick=()=>navigate('module',{id:m.id});
    grid.appendChild(cell);
  });
}

function renderModule(el,id){
  const m=MODULES.find(x=>x.id===id);if(!m)return navigate('dashboard');
  const done=S.completed.includes(id);
  const sections=[];
  if(m.hasReview)sections.push({name:'기억력 챌린지',icon:'⏱️',type:'review',reward:'30-50 FC',done:!!S.reviewsDone[id]});
  if(m.hasCode)sections.push({name:'코드 추적',icon:'💻',type:'codetrace',reward:'최대 30 FC',done:!!S.codeTracesDone[id]});
  if(m.hasQuiz)sections.push({name:m.mastery?'마스터리 체크포인트 퀴즈':'최적화 챌린지 퀴즈',icon:m.mastery?'⚔️':'📝',type:'quiz',reward:'가변 FC',done:S.quizScores[id]!=null});
  if(m.boss)sections.push({name:`보스전: ${BOSSES[m.boss].name}`,icon:'🐉',type:'boss',reward:'300 FC + SC',done:S.bossCleared.includes(m.boss)});

  el.innerHTML=`
    <button class="btn btn-sm btn-secondary back-btn" onclick="navigate('dashboard')">← 대시보드</button>
    <div class="card">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px">
        <div style="font-size:2rem;width:48px;height:48px;display:flex;align-items:center;justify-content:center;background:var(--card2);border-radius:12px">${m.level!=null?'L'+m.level:'📖'}</div>
        <div>
          <h2 style="margin:0">${m.title}</h2>
          <div style="color:var(--dim);font-size:.85rem">${m.sub}</div>
        </div>
      </div>
      ${m.level!=null?`<div style="font-size:.85rem;color:var(--dim);margin-bottom:12px">
        ${RANKS.find(r=>r.modules.includes(id))?`완료 시 승진 조건 진행`:''}</div>`:''}
      <div class="section-list" id="sectionList"></div>
    </div>
    ${done?`<div class="result-banner pass" style="margin-top:8px"><h2>✅ 모듈 완료</h2>
      <p>퀴즈 점수: ${S.quizScores[id]||0}%</p></div>`:''}`;

  const list=document.getElementById('sectionList');
  sections.forEach((s,i)=>{
    const prevDone=i===0||sections[i-1].done;
    const locked=!prevDone&&!s.done;
    const div=document.createElement('div');
    div.className=`section-item ${s.done?'done':''} ${locked?'locked':''}`;
    div.innerHTML=`
      <div class="sec-icon">${s.done?'✅':s.icon}</div>
      <div class="sec-name">${s.name}</div>
      <div class="sec-reward">${s.done?'완료':s.reward}</div>`;
    if(!locked)div.onclick=()=>{
      if(s.type==='quiz')navigate('quiz',{id});
      else if(s.type==='codetrace')navigate('codetrace',{id});
      else if(s.type==='review')navigate('review',{id});
      else if(s.type==='boss')navigate('boss',{id:m.boss});
    };
    list.appendChild(div);
  });
}

// === QUIZ (Game F) ===
function renderQuiz(el,modId){
  const qs=QUIZZES[modId];if(!qs)return navigate('module',{id:modId});
  const m=MODULES.find(x=>x.id===modId);
  const gflopsRange=m.level!=null?RANKS.find(r=>r.modules.includes(modId)):null;
  let idx=0,combo=0,totalFC=0,correct=0,answered=[];
  const hintsAvail=S.inventory.hint||0;

  function render(){
    const q=qs[idx];
    const pct=Math.round(correct/qs.length*100);
    const mult=getComboMult(combo);
    const gfStr=gflopsRange?`${Math.round(gflopsRange.gflops*pct/100).toLocaleString()}`:`—`;
    el.innerHTML=`
      <button class="btn btn-sm btn-secondary back-btn" onclick="navigate('module',{id:'${modId}'})">← 모듈 ${modId}</button>
      <div class="quiz-header">
        <span style="color:var(--dim)">문항 ${idx+1}/${qs.length}</span>
        <div class="combo-display ${mult>=3?'fire':''}">${combo>0?`${combo}연속 ×${mult}`:'콤보 없음'}</div>
        <span>FC: +${totalFC}</span>
      </div>
      <div class="gflops-meter">
        <div class="gflops-fill" style="width:${pct}%"></div>
        <div class="gflops-label">${gfStr} GFLOP/s (${pct}%)</div>
      </div>
      <div class="question-card">
        <span class="bloom-tag bloom-${q.b}">${BLOOM_ICONS[q.b]} ${BLOOM_LABELS[q.b]}</span>
        <div class="question-text">${q.q}</div>
        <div class="options" id="opts"></div>
        <div id="feedback"></div>
      </div>
      <div id="quizNav" style="display:none;text-align:center;margin-top:12px"></div>`;
    const opts=document.getElementById('opts');
    q.o.forEach((opt,oi)=>{
      const btn=document.createElement('button');
      btn.className='option-btn';
      btn.textContent=`(${String.fromCharCode(65+oi)}) ${opt}`;
      btn.onclick=()=>answer(oi);
      opts.appendChild(btn);
    });
  }

  function answer(chosen){
    const q=qs[idx];
    const isCorrect=chosen===q.a;
    const btns=document.querySelectorAll('.option-btn');
    btns.forEach((b,i)=>{
      b.disabled=true;
      if(i===q.a)b.classList.add('correct');
      if(i===chosen&&!isCorrect)b.classList.add('wrong');
    });
    if(isCorrect){
      combo++;
      const mult=getComboMult(combo);
      const fc=Math.round(20*mult);
      totalFC+=fc;correct++;
      addFC(fc);
      if(combo>S.maxCombo){S.maxCombo=combo;saveState();}
      showToast(`정답! +${fc} FC (×${mult})`,'reward');
    } else {
      combo=0;
    }
    answered.push(isCorrect);
    document.getElementById('feedback').innerHTML=`
      <div class="explanation">${isCorrect?'✅ 정답!':'❌ 오답'} — ${q.e}</div>`;
    const nav=document.getElementById('quizNav');
    nav.style.display='block';
    if(idx<qs.length-1){
      nav.innerHTML=`<button class="btn btn-primary" onclick="nextQ()">다음 문항 →</button>`;
    } else {
      nav.innerHTML=`<button class="btn btn-gold" onclick="finishQuiz()">결과 보기</button>`;
    }
    window.nextQ=()=>{idx++;render()};
    window.finishQuiz=()=>{
      const score=Math.round(correct/qs.length*100);
      if(score===100){totalFC+=100;addFC(100);}
      S.quizScores[modId]=Math.max(S.quizScores[modId]||0,score);
      S.quizAttempts[modId]=(S.quizAttempts[modId]||0)+1;
      if(score===100)S.perfectModules=(S.perfectModules||0)+1;
      const isMastery=MODULES.find(x=>x.id===modId)?.mastery;
      const passed=isMastery?score>=80:score>=60;
      if(passed&&!S.completed.includes(modId)){
        S.completed.push(modId);
        if(isMastery&&!S.masteryPassed.includes(modId))S.masteryPassed.push(modId);
        addFC(isMastery?300:0);
      }
      saveState();calcGflops();checkRankUp();checkAchievements();
      navigate('result',{modId,score,correct,total:qs.length,fc:totalFC,maxCombo:Math.max(...answered.map((_,i)=>{let c=0,m=0;for(let j=0;j<=i;j++){if(answered[j])c++;else c=0;if(c>m)m=c;}return m;})),passed,isMastery});
    };
  }
  render();
}

// === CODE TRACE (Game C) ===
function renderCodeTrace(el,modId){
  const ct=CODE_TRACES[modId];if(!ct)return navigate('module',{id:modId});
  let idx=0,streak=0,fc=0;

  function render(){
    const step=ct.steps[idx];
    el.innerHTML=`
      <button class="btn btn-sm btn-secondary back-btn" onclick="navigate('module',{id:'${modId}'})">← 모듈 ${modId}</button>
      <div class="card">
        <h2>💻 코드 추적: ${ct.title}</h2>
        <div class="code-block">${ct.code}</div>
      </div>
      <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
          <span>스텝 ${idx+1}/${ct.steps.length}</span>
          <span class="streak-display">${streak>=5?'🔥':''}연속 정답: ${streak}</span>
        </div>
        <div class="question-text">${step.prompt}</div>
        <div class="trace-input">
          <input type="number" id="traceAnswer" placeholder="답 입력" onkeydown="if(event.key==='Enter')checkTrace()">
          <button class="btn btn-primary btn-sm" onclick="checkTrace()">확인</button>
          <button class="btn btn-sm btn-secondary" onclick="showHintTrace()">힌트</button>
        </div>
        <div id="traceFeedback"></div>
        <div id="traceNav" style="display:none;margin-top:12px"></div>
      </div>
      <div style="text-align:right;color:var(--dim);font-size:.85rem">획득 FC: +${fc}</div>`;
    setTimeout(()=>{const inp=document.getElementById('traceAnswer');if(inp)inp.focus()},100);
  }

  window.showHintTrace=()=>{
    document.getElementById('traceFeedback').innerHTML=`<div class="explanation">💡 힌트: ${ct.steps[idx].hint}</div>`;
  };
  window.checkTrace=()=>{
    const inp=document.getElementById('traceAnswer');
    const val=parseInt(inp.value);
    const correct=val===ct.steps[idx].answer;
    if(correct){streak++;if(streak===5){fc+=30;addFC(30);showToast('5연속 정답! +30 FC','reward');}
      if(streak===10){fc+=30;addFC(30);showToast('10연속! +30 FC','reward');}
      document.getElementById('traceFeedback').innerHTML=`<div class="explanation" style="border-color:var(--green)">✅ 정답! ${ct.steps[idx].answer}</div>`;
    } else {
      streak=0;
      document.getElementById('traceFeedback').innerHTML=`<div class="explanation" style="border-color:var(--red)">❌ 오답. 정답: ${ct.steps[idx].answer}<br>💡 ${ct.steps[idx].hint}</div>`;
    }
    inp.disabled=true;
    const nav=document.getElementById('traceNav');nav.style.display='block';
    if(idx<ct.steps.length-1){
      nav.innerHTML=`<button class="btn btn-primary btn-sm" onclick="nextTrace()">다음 스텝 →</button>`;
    } else {
      nav.innerHTML=`<button class="btn btn-gold btn-sm" onclick="finishTrace()">완료</button>`;
    }
    window.nextTrace=()=>{idx++;render()};
    window.finishTrace=()=>{
      S.codeTracesDone[modId]=true;saveState();checkAchievements();
      showToast(`코드 추적 완료! +${fc} FC`,'reward');
      navigate('module',{id:modId});
    };
  };
  render();
}

// === MEMORY CHALLENGE (Game A) ===
function renderReview(el,modId){
  const qs=REVIEWS[modId];if(!qs)return navigate('module',{id:modId});
  let idx=0,correct=0,startTime=Date.now();
  const TIMEOUT=60000;

  function render(){
    const elapsed=Date.now()-startTime;
    const remaining=Math.max(0,Math.ceil((TIMEOUT-elapsed)/1000));
    const q=qs[idx];
    el.innerHTML=`
      <button class="btn btn-sm btn-secondary back-btn" onclick="navigate('module',{id:'${modId}'})">← 모듈 ${modId}</button>
      <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
          <h2>⏱️ 기억력 챌린지</h2>
          <div class="timer ${remaining<=30?remaining<=10?'danger':'warning':''}">${remaining}초</div>
        </div>
        <div style="color:var(--dim);font-size:.85rem;margin-bottom:12px">이전 모듈에서 배운 내용을 떠올려보세요! (${idx+1}/3)</div>
        <div class="question-text">${q.q}</div>
        <div class="options" id="revOpts"></div>
        <div id="revFeedback"></div>
      </div>`;
    const opts=document.getElementById('revOpts');
    q.o.forEach((opt,oi)=>{
      const btn=document.createElement('button');
      btn.className='option-btn';
      btn.textContent=opt;
      btn.onclick=()=>answerReview(oi);
      opts.appendChild(btn);
    });
  }

  function answerReview(chosen){
    const q=qs[idx];
    const isCorrect=chosen===q.a;
    if(isCorrect)correct++;
    document.querySelectorAll('.option-btn').forEach((b,i)=>{b.disabled=true;if(i===q.a)b.classList.add('correct');if(i===chosen&&!isCorrect)b.classList.add('wrong');});
    document.getElementById('revFeedback').innerHTML=`<div class="explanation">${isCorrect?'✅ 정답!':'❌ 오답'}</div>`;
    setTimeout(()=>{
      if(idx<2){idx++;render();}
      else{
        const elapsed=Date.now()-startTime;
        let fc=30;
        if(correct===3&&elapsed<30000){fc=50;showToast('30초 내 만점! +50 FC','reward');}
        else showToast(`기억력 챌린지 완료! +${fc} FC`,'reward');
        addFC(fc);S.reviewsDone[modId]=true;saveState();
        setTimeout(()=>navigate('module',{id:modId}),1000);
      }
    },800);
  }
  render();
  const timer=setInterval(()=>{
    const rem=Math.max(0,Math.ceil((TIMEOUT-(Date.now()-startTime))/1000));
    const td=document.querySelector('.timer');if(td){
      td.textContent=rem+'초';
      td.className=`timer ${rem<=10?'danger':rem<=30?'warning':''}`;
    }
    if(rem<=0){clearInterval(timer);addFC(30);S.reviewsDone[modId]=true;saveState();
      showToast('시간 초과! +30 FC (참여 보상)','reward');
      navigate('module',{id:modId});}
  },1000);
}

// === BOSS FIGHT ===
function renderBoss(el,bossId){
  const boss=BOSSES[bossId];if(!boss)return navigate('dashboard');
  let phase=0,correct=0;

  function render(){
    const p=boss.phases[phase];
    el.innerHTML=`
      <button class="btn btn-sm btn-secondary back-btn" onclick="navigate('module',{id:'${boss.module}'})">← 모듈 ${boss.module}</button>
      <div class="boss-header">
        <h2>⚔️ ${boss.name}</h2>
        <div class="boss-sub">마스터리 체크포인트 — 80% 이상 통과 필수</div>
        <div class="phase-indicator">${boss.phases.map((_,i)=>`<div class="phase-dot ${i<phase?'done':i===phase?'active':''}"></div>`).join('')}</div>
      </div>
      <div class="card">
        <h3>${p.title}</h3>
        <div class="question-text" style="margin:12px 0">${p.desc}</div>
        <div class="question-text" style="font-weight:600">${p.q}</div>
        <div class="options" id="bossOpts"></div>
        <div id="bossFeedback"></div>
      </div>`;
    const opts=document.getElementById('bossOpts');
    p.o.forEach((opt,oi)=>{
      const btn=document.createElement('button');
      btn.className='option-btn';
      btn.textContent=opt;
      btn.onclick=()=>answerBoss(oi);
      opts.appendChild(btn);
    });
  }

  function answerBoss(chosen){
    const p=boss.phases[phase];
    const isCorrect=chosen===p.a;
    if(isCorrect)correct++;
    document.querySelectorAll('.option-btn').forEach((b,i)=>{b.disabled=true;if(i===p.a)b.classList.add('correct');if(i===chosen&&!isCorrect)b.classList.add('wrong');});
    document.getElementById('bossFeedback').innerHTML=`<div class="explanation">${isCorrect?'✅ 정답!':'❌ 오답'} — ${p.e}</div>`;
    setTimeout(()=>{
      if(phase<boss.phases.length-1){phase++;render();}
      else finishBoss();
    },1500);
  }

  function finishBoss(){
    const score=Math.round(correct/boss.phases.length*100);
    const passed=score>=80;
    if(passed){
      addFC(300);
      const scAmt=correct===boss.phases.length?10:5;
      addSC(scAmt);
      if(!S.bossCleared.includes(bossId)){S.bossCleared.push(bossId);saveState();}
      if(!S.masteryPassed.includes(boss.module)){S.masteryPassed.push(boss.module);saveState();}
      if(!S.completed.includes(boss.module)){S.completed.push(boss.module);saveState();}
      checkRankUp();checkAchievements();
      showToast(`보스전 클리어! +300 FC +${correct===boss.phases.length?10:5} SC`,'rank');
    }
    navigate('result',{modId:boss.module,score,correct,total:boss.phases.length,fc:passed?300:0,passed,isMastery:true,isBoss:true,bossName:boss.name});
  }
  render();
}

// === RESULT ===
function renderResult(el,p){
  el.innerHTML=`
    <div class="result-banner ${p.passed?'pass':'fail'}">
      <div style="font-size:2.5rem;margin-bottom:8px">${p.passed?'🎉':'😓'}</div>
      <h2>${p.isBoss?p.bossName:p.isMastery?'마스터리 체크포인트':'퀴즈'} ${p.passed?'통과!':'미통과'}</h2>
      <p style="font-size:1.2rem;margin:8px 0">${p.score}%</p>
      ${!p.passed&&p.isMastery?'<p style="color:var(--dim)">80% 이상 필요합니다. 복습 후 재도전하세요!</p>':''}
    </div>
    <div class="stat-row">
      <div class="stat-box"><div class="stat-val">${p.correct}/${p.total}</div><div class="stat-label">정답</div></div>
      <div class="stat-box"><div class="stat-val" style="color:var(--cyan)">+${p.fc}</div><div class="stat-label">획득 FC</div></div>
      <div class="stat-box"><div class="stat-val" style="color:var(--green)">${S.gflops.toLocaleString()}</div><div class="stat-label">GFLOP/s</div></div>
    </div>
    <div style="text-align:center;margin-top:20px">
      <button class="btn btn-primary" onclick="navigate('module',{id:'${p.modId}'})">모듈로 돌아가기</button>
      <button class="btn btn-secondary" style="margin-left:8px" onclick="navigate('dashboard')">대시보드</button>
    </div>`;
}

// === SHOP ===
function renderShop(el){
  el.innerHTML=`
    <button class="btn btn-sm btn-secondary back-btn" onclick="navigate('dashboard')">← 대시보드</button>
    <div class="card">
      <h2>🛒 상점</h2>
      <div class="tabs">
        <div class="tab active" onclick="showShopTab('fc',this)">⚡ FC 상점</div>
        <div class="tab" onclick="showShopTab('sc',this)">💎 SC 상점</div>
      </div>
      <div id="shopContent"></div>
    </div>`;
  window.showShopTab=(type,tabEl)=>{
    document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
    tabEl.classList.add('active');
    const items=type==='fc'?SHOP_FC:SHOP_SC;
    const currency=type==='fc'?'FC':'SC';
    const bal=type==='fc'?S.fc:S.sc;
    const content=document.getElementById('shopContent');
    content.innerHTML=`<div style="text-align:right;margin-bottom:12px;font-weight:600">${type==='fc'?'⚡':'💎'}잔액: ${bal} ${currency}</div><div class="shop-grid"></div>`;
    const grid=content.querySelector('.shop-grid');
    items.forEach(item=>{
      const owned=S.inventory[item.id]||0;
      const maxed=item.max!=null&&owned>=item.max;
      const canBuy=!maxed&&(type==='fc'?S.fc:S.sc)>=item.price;
      const div=document.createElement('div');
      div.className='shop-item';
      div.innerHTML=`
        <div class="item-name">${item.icon} ${item.name}</div>
        <div class="item-desc">${item.desc}</div>
        ${maxed?'<div style="color:var(--dim);font-size:.85rem">구매 완료</div>':`
        <div style="display:flex;justify-content:space-between;align-items:center">
          <span class="item-price" style="color:${type==='fc'?'var(--cyan)':'var(--gold)'}">${type==='fc'?'⚡':'💎'}${item.price}</span>
          <button class="btn btn-sm ${canBuy?'btn-primary':'btn-secondary'}" ${canBuy?'':'disabled'} onclick="buyItem('${item.id}','${type}',${item.price})">구매${owned?' ('+owned+'개)':''}</button>
        </div>`}`;
      grid.appendChild(div);
    });
  };
  window.buyItem=(id,type,price)=>{
    const ok=type==='fc'?spendFC(price):spendSC(price);
    if(ok){S.inventory[id]=(S.inventory[id]||0)+1;saveState();showToast('구매 완료!','reward');showShopTab(type,document.querySelector('.tab.active'));}
  };
  showShopTab('fc',document.querySelector('.tab.active'));
}

// === BADGES ===
function renderBadges(el){
  el.innerHTML=`
    <button class="btn btn-sm btn-secondary back-btn" onclick="navigate('dashboard')">← 대시보드</button>
    <div class="card">
      <h2>🏆 업적 배지</h2>
      <div style="color:var(--dim);margin-bottom:16px">${S.earnedBadges.length}개 획득</div>
      <div class="badge-grid" id="badgeGrid"></div>
    </div>`;
  const grid=document.getElementById('badgeGrid');
  ACHIEVEMENTS.forEach(a=>{
    const earned=S.earnedBadges.includes(a.id);
    const show=!a.hidden||earned;
    if(!show)return;
    const div=document.createElement('div');
    div.className=`badge ${earned?'earned':'locked'}`;
    div.innerHTML=`<div class="badge-icon">${earned?a.icon:'❓'}</div><div>${earned?a.name:'???'}</div>
      <div style="color:var(--dim);font-size:.65rem">Tier ${a.tier}</div>`;
    grid.appendChild(div);
  });
}

// === INIT ===
function init(){
  const app=document.getElementById('app');
  app.innerHTML=`<nav id="topnav" style="display:none"></nav><main id="main"></main>`;
  if(S.name){
    document.getElementById('topnav').style.display='flex';
    updateNav();calcGflops();checkStreak();navigate('dashboard');
  } else {
    navigate('welcome');
  }
}
init();
